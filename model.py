import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffdim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4, groups=dim//4, kernel_size=3, padding='same')
        self.dwconv2 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4, groups=dim//4, kernel_size=7, padding='same')
        self.dwconv3 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4, groups=dim//4, kernel_size=7, dilation=3, padding='same')
        self.ln = LayerNorm2d(dim)
        self.fc1 = nn.Conv2d(in_channels=dim, out_channels=ffdim*2, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=ffdim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        xc = torch.chunk(x, 4, dim=1)
        x1 = self.dwconv1(xc[1])
        x2 = self.dwconv2(xc[2])
        x3 = F.avg_pool2d(xc[3], 3, stride=1, padding=1)
        x3 = self.dwconv3(x3)
        x = torch.cat([xc[0], x1, x2, x3], dim=1)
        x = self.ln(x)
        x = self.fc1(x)
        x = torch.chunk(x, 2, dim=1)
        x = F.gelu(x[0]) * x[1]
        x = self.fc2(x)
        return x


class Tinier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, padding=1)
        self.ln1 = LayerNorm2d(40)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, groups=40, stride=1, padding=1)
        self.l1 = Block(40, 72)
        self.l2 = Block(40, 72)
        self.dsconv = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=4, groups=20, stride=2, padding=1)
        self.ln2 = LayerNorm2d(80)
        self.l3 = Block(80, 144)
        self.l4 = Block(80, 144)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.conv2(x)
        x = self.l1(x)
        x = self.l2(x) + x
        x = self.dsconv(x)
        x = self.ln2(x)
        x = self.l3(x) + x
        x = self.l4(x) + x
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))
        return x
