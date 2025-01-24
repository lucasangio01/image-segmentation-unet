import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down_convolution_1 = DownSample(in_channels,
                                             64)  # each time, it goes through a double convolution and a max pooling
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.output_convolution = OutputConv(64, out_channels)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(
            x)  # there are two inputs: the downsample and the pooling for the skip connection
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.output_convolution(up_4)
        return out


class UNetPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetPP, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down_convolution_1 = NestedDoubleConv(in_channels, 64, 64)
        self.down_convolution_2 = NestedDoubleConv(64, 128, 128)
        self.down_convolution_3 = NestedDoubleConv(128, 256, 256)
        self.down_convolution_4 = NestedDoubleConv(256, 512, 512)

        self.bottle_neck = NestedDoubleConv(512, 1024, 1024)

        self.up_convolution_1 = NestedDoubleConv(64 + 128, 64, 64)
        self.up_convolution_2 = NestedDoubleConv(128 + 256, 128, 128)
        self.up_convolution_3 = NestedDoubleConv(256 + 512, 256, 256)
        self.up_convolution_4 = NestedDoubleConv(512 + 1024, 512, 512)

        self.up_convolution_5 = NestedDoubleConv(64 * 2 + 128, 64, 64)
        self.up_convolution_6 = NestedDoubleConv(128 * 2 + 256, 128, 128)
        self.up_convolution_7 = NestedDoubleConv(256 * 2 + 512, 256, 256)

        self.up_convolution_8 = NestedDoubleConv(64 * 3 + 128, 64, 64)
        self.up_convolution_9 = NestedDoubleConv(128 * 3 + 256, 128, 128)

        self.up_convolution_10 = NestedDoubleConv(64 * 4 + 128, 64, 64)

        self.output_convolution = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.down_convolution_2(self.pool(down_1))
        up_1 = self.up_convolution_1(torch.cat([down_1, self.Up(down_2)], 1))

        down_3 = self.down_convolution_3(self.pool(down_2))
        up_2 = self.up_convolution_2(torch.cat([down_2, self.Up(down_3)], 1))
        up_3 = self.up_convolution_5(torch.cat([down_1, up_1, self.Up(up_2)], 1))

        down_4 = self.down_convolution_4(self.pool(down_3))
        up_4 = self.up_convolution_3(torch.cat([down_3, self.Up(down_4)], 1))
        up_5 = self.up_convolution_6(torch.cat([down_2, up_2, self.Up(up_4)], 1))
        up_6 = self.up_convolution_8(torch.cat([down_1, up_1, up_3, self.Up(up_5)], 1))

        b = self.bottle_neck(self.pool(down_4))
        up_7 = self.up_convolution_4(torch.cat([down_4, self.Up(b)], 1))
        up_8 = self.up_convolution_7(torch.cat([down_3, up_4, self.Up(up_7)], 1))
        up_9 = self.up_convolution_9(torch.cat([down_2, up_2, up_5, self.Up(up_8)], 1))
        up_10 = self.up_convolution_10(torch.cat([down_1, up_1, up_3, up_6, self.Up(up_9)], 1))

        output = self.output_convolution(up_10)
        return output


class AttUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(AttUNet, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(in_channels, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = DoubleConv(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = DoubleConv(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = DoubleConv(128, 64)

        self.Conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pooled = self.pool(down)
        return down, pooled


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class NestedDoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(NestedDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class OutputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


'''
class InputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x
'''
