import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 坐标注意力 ==========
class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.pool_h(x)                     # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, W) -> (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)         # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_h = self.conv_h(y_h).sigmoid()          # (B, C, H, 1)
        y_w = self.conv_w(y_w).sigmoid()          # (B, C, W, 1)

        return x * y_h * y_w.permute(0, 1, 3, 2)


# ========== 渐进上采样解码器模块 ==========
class ProgressiveUpsampleDecoder(nn.Module):
    def __init__(self, in_channels_low, in_channels_skip, out_channels, up_scale=2):
        super().__init__()
        self.up_scale = up_scale
        self.pixel_shuffle_conv = nn.Conv2d(in_channels_low,
                                             out_channels * (up_scale ** 2),
                                             kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels + in_channels_skip, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.coord_att = CoordAtt(out_channels)

    def forward(self, low_feat, skip_feat):
        low_up = self.pixel_shuffle_conv(low_feat)
        low_up = self.pixel_shuffle(low_up)          # 尺寸与 skip_feat 一致
        fused = torch.cat([low_up, skip_feat], dim=1)
        fused = self.fuse_conv(fused)
        out = self.coord_att(fused)
        return out


# ========== TransUNet 三阶段渐进解码器 ==========
class TransUNetProgressiveDecoder(nn.Module):
    """
    解码器：接收 Transformer 输出（序列特征）和 CNN 编码器的三个跳跃连接特征，
    通过三个 ProgressiveUpsampleDecoder 逐级上采样并融合。
    """
    def __init__(self,
                 transformer_out_channels,   # Transformer 输出通道数
                 encoder_channels,           # 编码器各阶段通道数，从深到浅，[512, 256, 128]
                 decoder_channels,           # 解码器各阶段输出通道  [256, 128, 64]
                 patch_size=16,              # ViT 的 patch size，用于计算特征图尺寸
                 input_size=224):            # 输入图像尺寸
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        # 计算 Transformer 输出特征图的分辨率
        self.feat_h = self.feat_w = input_size // patch_size

        # 第一级：将 Transformer 输出 reshape 为特征图
        self.reshape_conv = nn.Conv2d(transformer_out_channels, transformer_out_channels, kernel_size=1)

        # 构建三个解码器阶段
        # 第1阶段：输入 Transformer 特征图，跳跃连接为 encoder_channels[0]（最深层）
        self.decoder1 = ProgressiveUpsampleDecoder(
            in_channels_low=transformer_out_channels,
            in_channels_skip=encoder_channels[0],
            out_channels=decoder_channels[0],
            up_scale=2
        )
        # 第2阶段：输入上一级输出，跳跃连接为 encoder_channels[1]
        self.decoder2 = ProgressiveUpsampleDecoder(
            in_channels_low=decoder_channels[0],
            in_channels_skip=encoder_channels[1],
            out_channels=decoder_channels[1],
            up_scale=2
        )
        # 第3阶段：输入上一级输出，跳跃连接为 encoder_channels[2]
        self.decoder3 = ProgressiveUpsampleDecoder(
            in_channels_low=decoder_channels[1],
            in_channels_skip=encoder_channels[2],
            out_channels=decoder_channels[2],
            up_scale=2
        )

        # 最终上采样到原图尺寸（如果需要）
        # 经过三次上采样后分辨率变为 input_size/(patch_size/8) = input_size/2 （若 patch_size=16，则变为 input_size/2）
        # 再上采样2倍恢复到原图
        self.final_upsample = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=1)
        )
        self.final_pixel_shuffle = nn.PixelShuffle(2)   # 2倍上采样到原图

    def forward(self, transformer_out, skip_feats):
        """
        transformer_out: (B, N, C)  ViT 输出序列
        skip_feats: list of (B, C, H, W)  从深到浅的编码器跳跃连接（三个）
        """
        B, N, C = transformer_out.shape
        # reshape 为特征图 (B, C, H, W)
        h = w = int(N ** 0.5)  # 假设 N 是平方数
        x = transformer_out.permute(0, 2, 1).view(B, C, h, w)
        x = self.reshape_conv(x)

        # 三个解码阶段
        d1 = self.decoder1(x, skip_feats[0])       # 输出尺寸: (2h, 2w)
        d2 = self.decoder2(d1, skip_feats[1])      # 输出尺寸: (4h, 4w)
        d3 = self.decoder3(d2, skip_feats[2])      # 输出尺寸: (8h, 8w)

        # 最终上采样到原图
        out = self.final_upsample(d3)
        out = self.final_pixel_shuffle(out)         # 输出尺寸: (16h, 16w) = (input_size, input_size)
        return out


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 假设参数
    transformer_out_channels = 768   # ViT-Base
    encoder_channels = [512, 256, 128]   # CNN 编码器三个阶段的通道数（从深到浅）
    decoder_channels = [256, 128, 64]    # 解码器各阶段输出通道数

    decoder = TransUNetProgressiveDecoder(
        transformer_out_channels=transformer_out_channels,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        patch_size=16,
        input_size=224
    )

    # 模拟输入
    transformer_out = torch.randn(2, 196, 768)   # 224/16=14, 14*14=196
    # 模拟三个跳跃连接（尺寸对应不同分辨率）
    skip_feats = [  
        torch.randn(2, 512, 14, 14),   # 最深层，14x14
        torch.randn(2, 256, 28, 28),   # 中层，28x28
        torch.randn(2, 128, 56, 56)    # 浅层，56x56
    ]

    output = decoder(transformer_out, skip_feats)
    print("Decoder output shape:", output.shape)  # (2, 64, 224, 224) 因为 final_upsample 输出通道为 decoder_channels[2]=64