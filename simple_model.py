#referenced from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


class CNN_from_mirex(nn.Module):
    #build model base on the model in MIREX 2020: Learning a Representation for Cover Song Identification Using Convolutional Neural Network
    #modify in_channels in self.conv0 and the output dimension in self.fc1 to suit other datasets
    def __init__(self,num_of_classes):
        super().__init__()
        #from documentation: nn.Conv2d(in_channels, out_channels, kernel_size,stride=1, padding=0, dilation=1, ......
        self.conv0 = nn.Conv2d(1,32,(12,3))
        self.conv0_bn = nn.BatchNorm2d(32, affine=False)
        self.conv1 = nn.Conv2d(32, 64, (13, 3), dilation = (1,2))
        self.conv1_bn = nn.BatchNorm2d(64, affine=False)
        self.maxpool = nn.MaxPool2d((1,2), (1,2))
        self.conv2 = nn.Conv2d(64, 64, (13,3))
        self.conv2_bn = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, 64, (3, 3),dilation = (1,2))
        self.conv3_bn = nn.BatchNorm2d(64, affine=False)
        self.conv4 = nn.Conv2d(64, 128, (3, 3))
        self.conv4_bn = nn.BatchNorm2d(128, affine=False)
        self.conv5 = nn.Conv2d(128, 128, (3, 3), dilation=(1, 2))
        self.conv5_bn = nn.BatchNorm2d(128, affine=False)
        self.conv6 = nn.Conv2d(128, 256, (3, 3))
        self.conv6_bn = nn.BatchNorm2d(256, affine=False)
        self.conv7 = nn.Conv2d(256, 256, (3, 3), dilation=(1, 2))
        self.conv7_bn = nn.BatchNorm2d(256, affine=False)
        self.conv8 = nn.Conv2d(256, 512, (3, 3))
        self.conv8_bn = nn.BatchNorm2d(512, affine=False)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), dilation=(1, 2))
        self.conv9_bn = nn.BatchNorm2d(512, affine=False)
        self.flatten = nn.Flatten()
        #modified from 1 x 1 x 512
        self.adaptivemaxpool = nn.AdaptiveMaxPool2d((1,1), return_indices=False)
        self.fc0 = nn.Linear(512,300)
        #8177 songs in shs100k-train, test dataset = 5 class
        self.fc1 = nn.Linear(300,num_of_classes)


    def forward(self, x):
        x = F.relu(self.conv0_bn(self.conv0(x)))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, (1,2),(1,2))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, (1, 2), (1, 2))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.max_pool2d(x, (1, 2), (1, 2))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.max_pool2d(x, (1, 2), (1, 2))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.relu(self.conv9_bn(self.conv9(x)))
        #note: following may be slightly different from the one in the paper
        x = self.adaptivemaxpool(x)
        x = self.flatten(x)
        #x = torch.reshape(x,(1,1,512))
        x =self.fc0(x)
        x =self.fc1(x)
        return x



