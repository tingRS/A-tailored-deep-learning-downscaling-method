import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels or in_channels
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)       
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class RCAB(nn.Module):
    def __init__(self, n_feats=64, reduction=16, res_scale=0.05, temp_init=2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(n_feats, n_feats // reduction, 1)
        self.fc2   = nn.Conv2d(n_feats // reduction, n_feats, 1)
        self.sig   = nn.Sigmoid()
        self.res_scale = res_scale
        self.temp = nn.Parameter(torch.tensor(float(temp_init)))
    def forward(self, x):
        y = self.conv1(x); y = self.relu(y); y = self.conv2(y)
        w = self.gap(y); w = self.relu(self.fc1(w)); w = self.sig(self.fc2(w))
        w = w / self.temp.clamp(min=1.0, max=5.0)  
        y = y * w
        return x + y * self.res_scale

class ResidualGroup(nn.Module):
    def __init__(self, n_feats=64, n_blocks=8, reduction=16, res_scale=0.05):
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(n_feats, reduction, res_scale) for _ in range(n_blocks)])
        self.tail   = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
    def forward(self, x):
        y = self.blocks(x)
        y = self.tail(y)
        return x + y

class Upsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super().__init__()
        layers = []
        num_ups = int(np.rint(np.log2(scale))) 
        for _ in range(num_ups):
            layers += [nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, stride=1, padding=1),nn.PixelShuffle(2),]
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        return self.body(x)

class RCAN(nn.Module):
    def __init__(self,number_channels=64,number_residual_blocks=8,upsampling_factor=8,dim=1,n_groups=5,reduction=16,res_scale=0.05):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True))
        self.groups = nn.Sequential(*[ResidualGroup(number_channels, number_residual_blocks, reduction, res_scale) for _ in range(n_groups)])
        self.body_tail = nn.Conv2d(number_channels, number_channels, 3, 1, 1)
        self.ups = Upsampler(upsampling_factor, number_channels)
        self.refine = nn.Sequential(nn.Conv2d(number_channels, number_channels, 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(number_channels, number_channels, 3, 1, 1), nn.ReLU(inplace=True))
        self.tail = nn.Conv2d(number_channels, dim, 1, 1, 0)
    def forward(self, x):
        h0 = self.head(x) 
        h  = self.groups(h0)
        h  = self.body_tail(h)
        h  = h + h0
        h  = self.ups(h)
        h  = self.refine(h)
        out = self.tail(h)
        return out

class ResNet(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=8, upsampling_factor=8, dim=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True))
        self.res_blocks = nn.ModuleList([ResidualBlock(number_channels, number_channels) for _ in range(number_residual_blocks)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
    def forward(self, x):
        out = self.conv1(x)
        for layer in self.upsampling:
            out = layer(out)
        out = self.conv2(out)
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


def gaussian_kernel(sigma: float):
    ks = int(2 * round(3 * float(sigma)) + 1)
    half = ks // 2
    xs = torch.arange(-half, half + 1, dtype=torch.float32)
    xx = xs.view(1, -1)   
    yy = xs.view(-1, 1) 
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.unsqueeze(0).unsqueeze(0)  
    return kernel, half


class RMResNet(nn.Module):
    def __init__(self,rcan: nn.Module,resnet: nn.Module,tau: float = 0.03,k: float = 4.0,feather_sigma: float = 0.2): 
        super().__init__()
        self.rcan = rcan
        self.resnet = resnet
        self.tau = float(tau)
        self.k   = float(k)
        kernel, pad = gaussian_kernel(float(feather_sigma))
        self.register_buffer('gauss_kernel', kernel)
        self.gauss_pad = pad
        self.last_mask = None
        self.last_mask_raw = None
    def _gaussian_blur(self, x):
        return F.conv2d(x, self.gauss_kernel, padding=self.gauss_pad, groups=1)
    def forward(self, x):
        y_rcan = self.rcan(x) 
        y_res  = self.resnet(x) 
        rcan_phys = torch.expm1(y_rcan) 
        res_phys  = torch.expm1(y_res)   
        m0 = torch.sigmoid(self.k * (rcan_phys - self.tau))  
        m_tilde = torch.clamp(self._gaussian_blur(m0), 0.0, 1.0)
        res_phys = torch.clamp(res_phys, min=0.0)
        y_phys = m_tilde * res_phys
        out = torch.log1p(y_phys)
        self.last_mask = m_tilde
        self.last_mask_raw = m0
        return out

