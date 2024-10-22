import torch.nn as nn
import torch
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING A FOOL MODEL ARCHITECTURE
# An instance of Net has been created in the model.py file
# 
######################################################################################

class Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        self.conv1 = nn.Conv2d(3, self.nb_channel, 5, padding='same')
        self.conv2 = nn.Conv2d(self.nb_channel, self.nb_channel, 5, padding='same')
        self.conv3 = nn.Conv2d(self.nb_channel, self.nb_channel, 5, padding='same')
        self.conv4 = nn.Conv2d(self.nb_channel, 1, 5, padding='same')
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
    

class UNet(nn.Module):
    def __init__(self, param):
        super(UNet, self).__init__()
        
        self.n_channels = param["MODEL"]["NB_CHANNEL"]
        self.img_size = param["DATASET"]["RESIZE_SHAPE"]

        # Encoder 
        self.encoder1 = self.encoder_block(3, self.n_channels)
        self.encoder2 = self.encoder_block(self.n_channels, self.n_channels * 2)
        self.encoder3 = self.encoder_block(self.n_channels * 2, self.n_channels * 4)
        self.encoder4 = self.encoder_block(self.n_channels * 4, self.n_channels * 8)

        self.bottleneck = self.encoder_block(self.n_channels * 8, self.n_channels * 16)

        # Decoder
        self.decoder4 = self.decoder_block(self.n_channels * 16  , self.n_channels * 8)
        self.decoder3 = self.decoder_block(self.n_channels * 8  , self.n_channels * 4)
        self.decoder2 = self.decoder_block(self.n_channels * 4 , self.n_channels * 2)
        self.decoder1 = self.decoder_block(self.n_channels * 2 , self.n_channels)


        self.final_conv = nn.Conv2d(self.n_channels, 3, kernel_size=1)

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
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder with skip connections
        dec4 = self.crop_and_concat(self.decoder4(bottleneck), enc4)
        dec3 = self.crop_and_concat(self.decoder3(dec4), enc3)
        dec2 = self.crop_and_concat(self.decoder2(dec3), enc2)
        dec1 = self.crop_and_concat(self.decoder1(dec2), enc1)

        # Final output layer
        return self.final_conv(dec1)






