import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


class FCN_VGG(nn.Module):
    def __init__(self, num_classes, downsample, upsample):
        super().__init__()

        self.upsample = upsample

        if downsample == 'vgg16':
            feats = list(models.vgg16(pretrained=True).features.children())
            self.cnn123 = nn.Sequential(*feats[0:17])
            self.cnn4 = nn.Sequential(*feats[17:24])
            self.cnn5 = nn.Sequential(*feats[24:])
        elif downsample == 'vgg19':
            feats = list(models.vgg19(pretrained=True).features.children())
            self.cnn123 = nn.Sequential(*feats[0:19])
            self.cnn4 = nn.Sequential(*feats[19:28])
            self.cnn5 = nn.Sequential(*feats[28:])

        # for i, f in enumerate(feats):
        #     print(i ,f)
        
        for m in self.parameters():
            m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.score_feat8 = nn.Conv2d(256, num_classes, 1)
        self.score_feat16 = nn.Conv2d(512, num_classes, 1)
        self.score_feat32 = nn.Conv2d(4096, num_classes, 1)

        if upsample == 'deconv':
            self.upsample32 = nn.ConvTranspose2d(num_classes, num_classes, stride=2, padding=1, kernel_size=3, output_padding=1)
            self.upsample16 = nn.ConvTranspose2d(num_classes, num_classes, stride=2, padding=1, kernel_size=3, output_padding=1)
            self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, stride=8, padding=1, kernel_size=9, output_padding=1)
        

    def forward(self, x):
        feat8 = self.cnn123(x)
        feat16 = self.cnn4(feat8)
        feat32 = self.fconn(self.cnn5(feat16))

        feat8 = self.score_feat8(feat8)
        feat16 = self.score_feat16(feat16)
        feat32 = self.score_feat32(feat32)

        # print('x', x.shape)
        # print('feat8', feat8.shape)
        # print('feat16', feat16.shape)
        # print('feat32', feat32.shape)

        if self.upsample == 'interploate':
            score = F.interpolate(feat32, feat16.size()[2:])
            score += feat16
            score = F.interpolate(score, feat8.size()[2:])
            score += feat8
            return F.interpolate(score, x.size()[2:])
        elif self.upsample == 'deconv':
            score = self.upsample32(feat32)
            score += feat16
            score = self.upsample16(score)
            score += feat8
            return self.upsample8(score)
