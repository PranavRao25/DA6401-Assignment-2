import torch
import torch.nn as nn
import yaml
from typing import Any
from pathlib import Path
from encoder import LazyVGG11Encode
import torch.nn.functional as F
from dropout import Dropout

"""
    INPUT: 256 * 256 * 3                 74 * CONV 1x1 LAYER 20 : 256 * 256 * 1
0 * CONV LAYER 1 : 256 * 256 * 64        73 * ReLU
1 * BATCHNORM                            72 * BATCHNORM
2 * ReLU ----------------------------->  71 * CONV LAYER 19 : 256 * 256 * 64
3 * MAXPOOL : 128 * 128 * 64             70 * ReLU
                                         69 * BATCHNORM
                                         68 * DECONV : 256 * 256 * 64
4 * CONV LAYER 2 : 128 * 128 * 128       67 * ReLU
5 * BATCHNORM                            66 * BATCHNORM
6 * ReLU ----------------------------->  65 * CONV LAYER 18 : 128 * 128 * 128
7 * MAXPOOL : 64 * 64 * 128              64 * ReLU                                          
                                         63 * BATCHNORM
                                         62 * DECONV : 128 * 128 * 128
8 * CONV LAYER 3 : 64 * 64 * 256         61 * ReLU
9 * BATCHNORM                            60 * BATCHNORM
10 * ReLU                                59 * CONV LAYER 17 : 64 * 64 * 256
11 * CONV LAYER 4 : 64 * 64 * 256        58 * ReLU
12 * BATCHNORM                           57 * BATCHNORM
13 * ReLU -----------------------------> 56 * CONV LAYER 16 : 64 * 64 * 256
14 * MAXPOOL : 32 * 32 * 256             55 * ReLU
                                         54 * BATCHNORM
                                         53 * DECONV : 64 * 64 * 256
15 * CONV LAYER 5 : 32 * 32 * 512        52 * ReLU
16 * BATCHNORM                           51 * BATCHNORM
17 * ReLU                                50 * CONV LAYER 15 : 32 * 32 * 512
18 * CONV LAYER 6 : 32 * 32 * 512        49 * ReLU
19 * BATCHNORM                           48 * BATCHNORM
20 * ReLU -----------------------------> 47 * CONV LAYER 14 : 32 * 32 * 512             
21 * MAXPOOL : 16 * 16 * 512             46 * ReLU
                                         45 * BATCHNORM
                                         44 * DECONV : 32 * 32 * 512
22 * CONV LAYER 7 : 16 * 16 * 512        43 * ReLU
23 * BATCHNORM                           42 * BATCHNORM
24 * ReLU                                41 * CONV LAYER 13 : 16 * 16 * 512
25 * CONV LAYER 8 : 16 * 16 * 512        40 * ReLU
26 * BATCHNORM                           39 * BATCHNORM
27 * ReLU -----------------------------> 38 * CONV LAYER 12 : 16 * 16 * 512
28 * MAXPOOL : 8  * 8  * 512             37 * ReLU
29 * CONV LAYER 10 : 8 * 8 * 512         36 * BATCHNORM 
30 * BATCHNORM                           35 * DECONV : 16 * 16 * 512 
31 * ReLU                                34 * ReLU
32 * CONV LAYER 11 : 8 * 8 * 512         33 * BATCHNORM

15s -2p + 2d + o = 31
s = 1, p = 0, k = 3, d = 1 => o = 6
s = 2, p = 1, k = 3, d = 1 => o = 1
"""

class LazyVGG11Upsample(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.deconv1 = nn.Sequential(
            nn.LazyConvTranspose2d(512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.LazyConvTranspose2d(512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.LazyConvTranspose2d(256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.LazyConvTranspose2d(128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.deconv5 = nn.Sequential(
            nn.LazyConvTranspose2d(64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.deconv_layers = [
            self.deconv1,
            self.conv1,
            self.conv2,
            self.deconv2,
            self.conv3,
            self.conv4,
            self.deconv3,
            self.conv5,
            self.conv6,
            self.deconv4,
            self.conv7,
            self.deconv5,
            self.conv8
        ]
    
    def forward(self, x: torch.Tensor):
        for layer in self.deconv_layers:
            x = layer(x)
        return x



class UNet(nn.Module):
    def __init__(self, no_classes=37, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = LazyVGG11Encode()

        self.intermediate_layer = nn.Sequential(
            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.LazyConv2d(512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.decoder = LazyVGG11Upsample()

        self.output = nn.Sequential(
            nn.LazyConv2d(no_classes, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _pad_concat(self, skip: torch.Tensor, upsampled: torch.Tensor):
        diffY = skip.size()[2] - upsampled.size()[2]
        diffX = skip.size()[3] - upsampled.size()[3]

        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX //2,
                                              diffX // 2, diffX - diffX //2])
        
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x: torch.Tensor):
        s1 = self.encoder.conv1(x)
        x = self.encoder.maxpool1(s1)

        s2 = self.encoder.conv2(x)
        x = self.encoder.maxpool2(x)

        x = self.encoder.conv3(x)
        s4 = self.encoder.conv4(x)
        x = self.encoder.maxpool3(s4)

        x = self.encoder.conv5(x)
        s6 = self.encoder.conv6(x)
        x = self.encoder.maxpool4(s6)

        x = self.encoder.conv7(x)
        s8 = self.encoder.conv8(x)
        x = self.encoder.maxpool5(x)

        x = self.intermediate_layer(x)

        x = self.decoder.deconv1(x)
        x = self._pad_concat(s8, x)
        x = self.decoder.conv1(x)
        x = self.decoder.conv2(x)

        x = self.decoder.deconv2(x)
        x = self._pad_concat(s6, x)
        x = self.decoder.conv3(x)
        x = self.decoder.conv4(x)

        x = self.decoder.deconv3(x)
        x = self._pad_concat(s4, x)
        x = self.decoder.conv5(x)
        x = self.decoder.conv6(x)

        x = self.decoder.deconv4(x)
        x = self._pad_concat(s2, x)
        x = self.decoder.conv7(x)

        x = self.decoder.deconv5(x)
        x = self._pad_concat(s1, x)
        x = self.decoder.conv8(x)

        x = self.output(x)
        
        return x