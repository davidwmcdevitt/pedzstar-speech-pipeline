import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        conv_layers = []
        
        def add_conv_block(in_channels, out_channels, kernel_size, stride, padding, use_maxpool=False, a=0.1):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            relu = nn.ReLU()
            bn = nn.BatchNorm2d(out_channels)
            init.kaiming_normal_(conv.weight, a=a)
            conv.bias.data.zero_()
            block = [conv, relu, bn]
            if use_maxpool:
                block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return block

        conv_layers += add_conv_block(2, 16, (5, 5), (2, 2), (2, 2))
        conv_layers += add_conv_block(16, 32, (3, 3), (2, 2), (1, 1))
        conv_layers += add_conv_block(32, 64, (3, 3), (1, 1), (1, 1))
        conv_layers += add_conv_block(64, 128, (3, 3), (1, 1), (1, 1))
        conv_layers += add_conv_block(128, 256, (3, 3), (2, 2), (1, 1))
        conv_layers += add_conv_block(256, 512, (3, 3), (2, 2), (1, 1))

        self.lin1 = nn.Linear(in_features=512, out_features=128)
        self.drop1 = nn.Dropout(0.1)
        self.lin2 = nn.Linear(in_features=128, out_features=32)
        self.drop2 = nn.Dropout(0.1)
        self.lin3 = nn.Linear(in_features=32, out_features=num_classes)

        self.lin = nn.Sequential(*[self.lin1, self.drop1, self.lin2, self.drop2, self.lin3])
        
        self.conv = nn.Sequential(*conv_layers)
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        
        x = self.conv(x)
        
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        
        x = self.lin(x)
        
        return F.softmax(x, dim=1)
    
    