""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_PAD = 'valid' # was 1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=CONV_PAD, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=CONV_PAD, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, decoder_layer, encoder_layer):
        # print(decoder_layer.size())
        decoder_layer = self.upsample(decoder_layer)
        # print(decoder_layer.size())
        # input is CHW
        # gresit!!
        diffY = encoder_layer.size()[2] - decoder_layer.size()[2]
        diffX = encoder_layer.size()[3] - decoder_layer.size()[3]
        # decoder_layer = F.pad(decoder_layer, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # print(f"diff : {diffY}, {diffX}")
        # crop encoder layer
        # print(encoder_layer.size())
        encoder_layer = encoder_layer[:,:,:-diffY,:-diffX]
        # print(encoder_layer.size())
        x = torch.cat([encoder_layer, decoder_layer], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetValid(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=False):
        super(UNetValid, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv0 = DoubleConv(in_channels, 64)
        self.down_conv1 = DownConv(64, 128)
        self.down_conv2 = DownConv(128, 256)
        self.down_conv3 = DownConv(256, 512)
        factor = 2 if bilinear else 1
        self.down_conv4 = DownConv(512, 1024 // factor)
        self.up_conv1 = UpConv(1024, 512 // factor, bilinear)
        self.up_conv2 = UpConv(512, 256 // factor, bilinear)
        self.up_conv3 = UpConv(256, 128 // factor, bilinear)
        self.up_conv4 = UpConv(128, 64, bilinear)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.conv0(x)
        # print(x1.shape)
        x2 = self.down_conv1(x1)
        # print(x2.shape)
        x3 = self.down_conv2(x2)
        # print(x3.shape)
        x4 = self.down_conv3(x3)
        # print(x4.shape)
        x5 = self.down_conv4(x4)
        # print(x5.shape)
        x = self.up_conv1(x5, x4)
        # print(x.shape)
        x = self.up_conv2(x, x3)
        # print(x.shape)
        x = self.up_conv3(x, x2)
        # print(x.shape)
        x = self.up_conv4(x, x1)
        # print(x.shape)
        logits = self.out_conv(x)
        # print(x.shape)
        return logits

    def use_checkpointing(self):
        self.conv0 = torch.utils.checkpoint(self.conv0)
        self.down_conv1 = torch.utils.checkpoint(self.down_conv1)
        self.down_conv2 = torch.utils.checkpoint(self.down_conv2)
        self.down_conv3 = torch.utils.checkpoint(self.down_conv3)
        self.down_conv4 = torch.utils.checkpoint(self.down_conv4)
        self.up_conv1 = torch.utils.checkpoint(self.up_conv1)
        self.up_conv2 = torch.utils.checkpoint(self.up_conv2)
        self.up_conv3 = torch.utils.checkpoint(self.up_conv3)
        self.up_conv4 = torch.utils.checkpoint(self.up_conv4)
        self.out_conv = torch.utils.checkpoint(self.out_conv)


if __name__ == "__main__":
    x = torch.randn((1, 3, 512, 512))
    model = UNetValid(in_channels=3, n_classes=1, bilinear=True)
    preds = model(x)
    # print(preds.shape)
    # print(x.shape)
