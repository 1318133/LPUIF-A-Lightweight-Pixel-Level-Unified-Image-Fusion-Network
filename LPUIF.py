import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvLea(nn.Module):
    """(convolution => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvLea1(nn.Module):
    """(convolution => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Convtanh(nn.Module):
    """(convolution => Tanh) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.double_conv(x)


def Concat(x, y, z):
    return torch.cat((x, y), z)

class wtNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pthfile = 'your file'):
        super(wtNet, self).__init__()
         
        # net= UNet(n_channels=1, n_classes=1, bilinear=True)
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear 

        self.xin1 = ConvLea(n_channels, 32)
        self.xin2 = ConvLea(32, 64)
        self.wei_xin = ConvLea1(64, 1)


        self.fon33 = ConvLea(128, 64)
        self.fon44 = Convtanh(64, 1)

        if pthfile is not None:
            # self.load_state_dict(torch.save(torch.load(pthfile), pthfile,_use_new_zipfile_serialization=False), strict = False)  # 训练所有数据后，保存网络的参数
            self.load_state_dict(torch.load(pthfile), strict = False)
        


    def forward(self, x, y):
        
        v1 = self.xin1(x) #32
        v2 = self.xin2(v1) #64


        r1 = self.xin1(y)
        r2 = self.xin2(r1)

        f1 = Concat(v2, r2, 1) #128

        v22 = self.wei_xin(v2)
        r22 = self.wei_xin(r2)

        f5 = self.fon33(f1)
        f6 = self.fon44(f5)

        return f6, f6, f6, f6, r2, v22, r22