import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=35):
        super(VGGFeatureExtractor, self).__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:layer_index])
        
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.criterion(sr_features, hr_features)

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, real_pred, fake_pred, is_discriminator=True):
        real_mean = torch.mean(real_pred)
        fake_mean = torch.mean(fake_pred)
        
        if is_discriminator:
            loss_real = self.bce(real_pred - fake_mean, torch.ones_like(real_pred))
            loss_fake = self.bce(fake_pred - real_mean, torch.zeros_like(fake_pred))
        else:
            loss_real = self.bce(real_pred - fake_mean, torch.zeros_like(real_pred))
            loss_fake = self.bce(fake_pred - real_mean, torch.ones_like(fake_pred))
        
        return (loss_real + loss_fake) / 2

class ESRGANLoss(nn.Module):
    def __init__(self, lambda_pixel=0.01, lambda_perceptual=1.0, lambda_adv=0.005):
        super(ESRGANLoss, self).__init__()
        
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv

    def generator_loss(self, sr, hr, real_pred, fake_pred):
        l_pixel = self.pixel_loss(sr, hr)
        l_perceptual = self.perceptual_loss(sr, hr)
        l_adv = self.adversarial_loss(real_pred, fake_pred, is_discriminator=False)
        
        total = (self.lambda_pixel * l_pixel + 
                 self.lambda_perceptual * l_perceptual + 
                 self.lambda_adv * l_adv)
        
        return total, {
            'pixel': l_pixel.item(),
            'perceptual': l_perceptual.item(),
            'adversarial': l_adv.item(),
            'total': total.item()
        }

    def discriminator_loss(self, real_pred, fake_pred):
        return self.adversarial_loss(real_pred, fake_pred, is_discriminator=True)

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        return self.criterion(sr, hr)
