from turtle import forward
import torch 
import torch.nn as nn
import cv2
import torch.nn.functional as F
import torch.nn as nn

class residualBlockShort(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, stride:int = 1, downsample:bool = False) -> None:
        super(residualBlockShort, self).__init__()
        self.__expansion = 1
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
        )
        if downsample == False:
            self.second_branch = nn.Sequential(
                nn.Identity(),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.second_branch = nn.Sequential(
                nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 1, stride = stride, padding = 0, bias = False),
                nn.BatchNorm2d(out_channel)
            )
        self.ReluE = nn.ReLU(inplace=True)
    def forward(self, x):
        x_main = self.main_branch(x)
        x_second = self.second_branch(x)
        x = x_main + x_second
        x = self.ReluE(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = False) -> None:
        super(ResNet, self).__init__()                                                                                                # input[3, 224, 224]
        self.comPart = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False),                    # output[64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)                                                                    # output[64, 56, 56]
        )
        self.conv2x = nn.Sequential(
            residualBlockShort(in_channel = 64, out_channel = 64, downsample = False),                                                # output[64, 56, 56] 
            residualBlockShort(in_channel = 64, out_channel = 64, downsample = False),                                                # output[64, 56, 56]
            residualBlockShort(in_channel = 64, out_channel = 64, downsample = False),                                                # output[64, 56, 56]
        )
        self.conv3x = nn.Sequential(
            residualBlockShort(in_channel = 64, out_channel = 128, stride = 2, downsample = True),                                    # output[256, 28, 28]
            residualBlockShort(in_channel = 128, out_channel = 128, downsample = False),                                              # output[256, 28, 28]
            residualBlockShort(in_channel = 128, out_channel = 128, downsample = False),                                              # output[256, 28, 28]
            residualBlockShort(in_channel = 128, out_channel = 128, downsample = False),                                              # output[256, 28, 28]
        )
        self.conv4x = nn.Sequential(
            residualBlockShort(in_channel = 128, out_channel = 256, stride = 2, downsample = True),                                   # output[256, 14, 14]
            residualBlockShort(in_channel = 256, out_channel = 256, downsample = False),                                              # output[256, 14, 14]
            residualBlockShort(in_channel = 256, out_channel = 256, downsample = False),                                              # output[256, 14, 14]
            residualBlockShort(in_channel = 256, out_channel = 256, downsample = False),                                              # output[256, 14, 14]
            residualBlockShort(in_channel = 256, out_channel = 256, downsample = False),                                              # output[256, 14, 14]
            residualBlockShort(in_channel = 256, out_channel = 256, downsample = False),                                              # output[256, 14, 14]
        )
        self.conv5x = nn.Sequential(
            residualBlockShort(in_channel = 256, out_channel = 512, stride = 2, downsample = True),                                   # output[512, 7, 7]
            residualBlockShort(in_channel = 512, out_channel = 512, downsample = False),                                              # output[512, 7, 7]
            residualBlockShort(in_channel = 512, out_channel = 512, downsample = False),                                              # output[512, 7, 7]
            nn.AdaptiveAvgPool2d((1,1))                                                                                               # output[512, 1, 1]
        )
        self.classifier = nn.Linear(512, num_classes)                                                                                 # output[1000]

        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.comPart(x)
        x = self.conv2x(x)
        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.conv5x(x)
        x = x.view(-1,1*1*512)
        x = self.classifier(x)
        return x

    # 用于初始化权重函数
    def _initialize_weights(self):
        # modules()方法来自nn.Module父类，可以读取当前网络所有结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 初始化均值为零，方差为0.01，偏置为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = False):
        super(AlexNet, self).__init__()
        # [in_channels, in_channels, kernel_size, stride, padding]
        self.features = nn.Sequential(                   # input[3,224,224]
            # 左侧1列0，右侧2列0，上方1行0，下方1行0       
            # nn.zeropad2d(padding=(1,2,1,2)),
            nn.Conv2d(3, 48, 11, 4, 2),                  # output[48, 55, 55]
            # inplace即增加计算量，换取更小的内存
            nn.ReLU(inplace=True),                      
            nn.MaxPool2d(3, 2),                          # output[48, 27, 27]
            nn.Conv2d(48, 128, 5, 1, 2),                 # output[128, 27, 27]
            nn.ReLU(inplace=True),                      
            nn.MaxPool2d(3, 2),                          # output[128, 13, 13]
            nn.Conv2d(128, 192, 3, 1, 1),                # output[192, 13, 13]
            nn.ReLU(inplace=True),                      
            nn.Conv2d(192, 192, 3, 1, 1),                # output[192, 13, 13]
            nn.ReLU(inplace=True),                      
            nn.Conv2d(192, 128, 3, 1, 1),                # output[128, 13, 13]
            nn.ReLU(inplace=True),                      
            nn.MaxPool2d(3, 2)                           # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            # 在pytorch中使用Dropout方法
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        # 通过flatten平铺特征向量
        # x = torch.flatten(x, start_dim=1)
        # 或者通过使用view方法
        x = x.view(-1,6*6*128)
        x = self.classifier(x)
        return x

    # 用于初始化权重函数
    def _initialize_weights(self):
        # modules()方法来自nn.Module父类，可以读取当前网络所有结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 初始化均值为零，方差为0.01，偏置为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VGG16(nn.Module):
    def __init__(self, num_classes = 1000, init_weights = False):
        super(VGG16, self).__init__()        
        # [in_channels, in_channels, kernel_size, stride, padding]       
        self.features = nn.Sequential(              # input[3,224,224]
            nn.Conv2d(3, 64, 3, 1, 1),              # output[64,224,224]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),             # output[64,224,224]  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # output[64,112,112]
            nn.Conv2d(64, 128, 3, 1, 1),            # output[128,112,112]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),           # output[128,112,112]  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # output[128,56,56]
            nn.Conv2d(128, 256, 3, 1, 1),           # output[256,56,56]
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),           # output[256,56,56]
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),           # output[256,56,56]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),           # output[256,56,56]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # output[256,28,28]
            nn.Conv2d(256, 512, 3, 1, 1),           # output[512,28,28]
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, 1, 1),           # output[512,28,28]
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),           # output[512,28,28]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),           # output[512,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # output[512,14,14]
            # nn.Conv2d(512, 512, 3, 1, 1),           # output[512,14,14]
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),           # output[512,14,14]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),           # output[512,14,14]
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),           # output[512,14,14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)                      # output[512,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,512*7*7)
        x = self.classifier(x)
        return x

    # 用于初始化权重函数
    def _initialize_weights(self):
        # modules()方法来自nn.Module父类，可以读取当前网络所有结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 初始化均值为零，方差为0.01，偏置为0
                nn.init.normal_(m.weight, 0, 0.002)
                nn.init.constant_(m.bias, 0)