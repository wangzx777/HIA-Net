import torch.nn as nn
import torch


def basic_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResCNN(nn.Module):
    '''
    Feature Extraction Model
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    the output of the backbone is a flatten feature vector, not softmax probs
    '''
    def __init__(self, input_channels=1, hid_channels=64, output_channels=64, output_dim=256):
        super(ResCNN, self).__init__()
        # define ProtoNet
        self.rb1 = nn.Sequential(
            basic_block(input_channels, hid_channels),
            basic_block(hid_channels, hid_channels)
        )
        self.res1 = basic_block(input_channels, hid_channels)
        self.bn1 = nn.BatchNorm2d(hid_channels)

        self.rb2 = nn.Sequential(
            basic_block(hid_channels, hid_channels),
            basic_block(hid_channels, hid_channels)
        )
        self.res2 = basic_block(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        #
        # self.rb3 = nn.Sequential(
        #     basic_block(hid_channels, hid_channels),
        #     basic_block(hid_channels, output_channels)
        # )
        # self.res3 = basic_block(hid_channels, output_channels)
        # self.bn3 = nn.BatchNorm2d(output_channels)


        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(512, output_dim)


    def forward(self, x):
        # x = torch.reshape(x, (-1, 1, 64, 64))  # MMID reshape data to 2D
        # x = torch.reshape(x, (-1, 1, 200, 62))  # SEED
        x = torch.reshape(x, (-1, 1, 32, 4))  # DEAP reshape data to 2D, 32chls*4bands

        x_res = self.res1(x)
        x = self.rb1(x)
        x += x_res
        x = self.bn1(x)
        x = self.maxpool(x)

        x_res = self.res2(x)
        x = self.rb2(x)
        x += x_res
        x = self.bn2(x)
        x = self.maxpool(x)
        #
        # x_res = self.res3(x)
        # x = self.rb3(x)
        # x += x_res
        # x = self.bn3(x)
        # x = self.maxpool(x)

        x = x.view(x.size(0), -1)  # flatten feature maps to a single feature vector
        output = self.linear(x)
        return output

class ResCBAM(nn.Module):
    '''
    Feature Extraction Model
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    the output of the backbone is a flatten feature vector, not softmax probs
    '''
    def __init__(self, input_channels=4, hid_channels=64, output_channels=64, output_dim=256):
        super(ResCBAM, self).__init__()
        self.input_channels = input_channels
        self.rb1 = nn.Sequential(
            basic_block(input_channels, hid_channels),
            basic_block(hid_channels, hid_channels)
        )
        self.res1 = basic_block(input_channels, hid_channels)

        self.ca1 = ChannelAttention(hid_channels)
        self.sa1 = SpatialAttention()

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(1024, output_dim)
        # self.out = nn.Linear(output_dim, 2)

    def forward(self, x):
        assert x.shape[1] == self.input_channels

        # 保留原输入，避免原地修改
        x_res = self.res1(x)
        x_rb1 = self.rb1(x)
        x_sum = x_rb1 + x_res

        # 通道注意力和空间注意力模块
        x_ca1 = self.ca1(x_sum) * x_sum
        x_sa1 = self.sa1(x_ca1) * x_ca1

        # 批标准化和池化
        x_bn1 = self.bn1(x_sa1)
        x_pooled = self.maxpool(x_bn1)

        # 将特征图展平为单个特征向量
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        output = self.linear(x_flat)

        return output
