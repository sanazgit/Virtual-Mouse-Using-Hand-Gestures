import torch.nn as nn

def conv3x3(in_channels, out_channels):
      """3x3 convolution with padding"""
      module = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1 , padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU())
      return module

def conv3x3_drop(in_channels, out_channels):
      """3x3 convolution with padding"""
      module = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1 , padding=1),
                             nn.Dropout(0.2),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU())
      return module
  
class CNNModel(nn.Module):
    def __init__(self, input_channel, num_cls):
        super().__init__()

        self.conv= nn.Sequential(conv3x3(input_channel, 75),
                                  nn.MaxPool2d(2,2), # 14x14

                                  conv3x3(75, 50),
                                  nn.MaxPool2d(2, 2), # x8

                                  conv3x3_drop(50, 25),
                                  nn.MaxPool2d(2, 2), # 4x4

                                  conv3x3(25, 512),

                                  nn.AdaptiveAvgPool2d(output_size=(1, 1)), # 1x1

                                  nn.Flatten(),
                                  nn.Linear(512, num_cls) # classifier
                          )


    def forward(self, x):

        y= self.conv(x)

        return y
