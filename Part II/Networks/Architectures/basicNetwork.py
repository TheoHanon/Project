import torch.nn as nn
import torch
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING A FOOL MODEL ARCHITECTURE
# An instance of Net has been created in the model.py file
# 
######################################################################################


class UNetLight(nn.Module):


    def __init__(self, param):
        super(UNetLight, self).__init__()
        self.n_channels = param["MODEL"]["NB_CHANNEL"]

        self.encoder1 = self.encoder_block(3, self.n_channels)
        self.encoder2 = self.encoder_block(self.n_channels, self.n_channels * 2)

        self.bottleneck = self.encoder_block(self.n_channels * 2, self.n_channels * 4)

        self.dropout2 = nn.Dropout2d(p=0.25)
        self.decoder2 = self.decoder_block(self.n_channels * 4 , self.n_channels * 2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.decoder1 = self.decoder_block(2*self.n_channels * 2 , self.n_channels)

        self.final_conv = nn.Conv2d(2*self.n_channels, 1, kernel_size=1)


    def encoder_block(self, in_channels, out_channels):
        """Two convolutional layers with batch normalization and ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def decoder_block(self, in_channels, out_channels):
        """Upsampling block using ConvTranspose followed by two Conv2D layers."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def crop_and_concat(self, upsampled, bypass):
        """Center-crop the bypass tensor to match the upsampled tensor."""
        bypass = F.interpolate(bypass, size=upsampled.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encoder
    
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc2, kernel_size=2))
        
        # Decoder with skip connections
        dec2 = self.crop_and_concat(self.decoder2(bottleneck), self.dropout2(enc2))
        dec1 = self.crop_and_concat(self.decoder1(dec2), self.dropout1(enc1))
    
        # Final output layer
        return self.final_conv(dec1)




