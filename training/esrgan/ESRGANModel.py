import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32):
        super(DenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32):
        super(RRDB, self).__init__()
        self.dense1 = DenseBlock(in_channels, growth_rate)
        self.dense2 = DenseBlock(in_channels, growth_rate)
        self.dense3 = DenseBlock(in_channels, growth_rate)

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, 
                 num_blocks=8, growth_rate=32, scale_factor=4):
        super(RRDBNet, self).__init__()
        
        self.scale_factor = scale_factor
        
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        self.trunk = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(num_blocks)]
        )
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        self.upconv1 = nn.Conv2d(num_features, num_features * 4, 3, 1, 1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        
        self.upconv2 = nn.Conv2d(num_features, num_features * 4, 3, 1, 1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        
        self.hr_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        for m in [self.conv_first, self.trunk_conv, self.upconv1, self.upconv2, 
                  self.hr_conv, self.conv_last]:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk
        
        feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))
        
        out = self.conv_last(self.lrelu(self.hr_conv(feat)))
        return out

class VGGDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(VGGDiscriminator, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features, num_features, 4, 2, 1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_features * 8, num_features * 8, 4, 2, 1),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    gen = RRDBNet(num_blocks=8).to(device)
    lr_img = torch.randn(1, 3, 64, 64).to(device)
    sr_img = gen(lr_img)
    print(f"Generator: LR {lr_img.shape} -> SR {sr_img.shape}")
    
    gen_params = sum(p.numel() for p in gen.parameters())
    print(f"Generator parameters: {gen_params:,}")
    
    disc = VGGDiscriminator().to(device)
    score = disc(sr_img)
    print(f"Discriminator: Input {sr_img.shape} -> Score {score.shape}")
    
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator parameters: {disc_params:,}")
