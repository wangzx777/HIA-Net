import torch.nn as nn
import torch


def basic_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    基础卷积块：Conv2d + BN + ReLU
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    
    结构：
    - Conv2d: 3x3卷积，padding=1保持空间尺寸
    - BatchNorm2d: 批归一化，加速训练
    - ReLU: 激活函数
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )


class ChannelAttention(nn.Module):
    """
    通道注意力模块 - CBAM的一部分
    
    功能：关注"什么"特征是有意义的
    通过平均池化和最大池化聚合空间信息，然后通过共享MLP学习通道权重
    """
    def __init__(self, in_planes, ratio=2):
        """
        Args:
            in_planes: 输入通道数
            ratio: 压缩比率，默认2，即通道数压缩为原来的1/2
        
        结构：
        - 平均池化和最大池化并行
        - 共享MLP: Conv1x1(压缩) -> ReLU -> Conv1x1(恢复)
        - 逐元素相加后经过Sigmoid得到通道权重
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # 共享MLP，用1x1卷积实现
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 压缩
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 恢复

        self.sigmoid = nn.Sigmoid()  # 输出0-1的权重

    def forward(self, x):
        """
        Args:
            x: 输入特征图，形状 (batch_size, channels, height, width)
        
        Returns:
            通道注意力权重，形状 (batch_size, channels, 1, 1)
        """
        # 平均池化分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # (batch, channels, 1, 1)
        
        # 最大池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # (batch, channels, 1, 1)
        
        # 融合两个分支
        out = avg_out + max_out
        return self.sigmoid(out)  # 返回通道权重


class SpatialAttention(nn.Module):
    """
    空间注意力模块 - CBAM的一部分
    
    功能：关注"哪里"是有意义的特征
    在通道维度上做平均和最大池化，然后通过卷积学习空间权重
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        # 输入2通道（平均池化和最大池化结果），输出1通道空间注意力图
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图，形状 (batch_size, channels, height, width)
        
        Returns:
            空间注意力权重，形状 (batch_size, 1, height, width)
        """
        # 在通道维度上做平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, h, w)
        
        # 在通道维度上做最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, h, w)
        
        # 拼接两个池化结果
        x = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, h, w)
        
        # 卷积学习空间权重
        x = self.conv1(x)  # (batch, 1, h, w)
        return self.sigmoid(x)


class ResCBAM(nn.Module):
    '''
    Feature Extraction Model
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    the output of the backbone is a flatten feature vector, not softmax probs
    
    EEG编码器 - 对应论文2.B.1节 EEG Encoder
    
    论文原文：
    "HIA-Net employs a convolutional neural network combining a residual block [27] 
     with a Convolutional Block Attention Module (CBAM) block [28] for EEG feature 
     extraction. This residual block comprises three convolutional layers with batch 
     normalization, integrating channel and spatial attention mechanisms via the CBAM 
     module."
    
    输入处理：
    - 论文中EEG信号被提取为差分熵特征，每个频带转换为9×9的2D矩阵
    - 输入通道数应该是5（对应5个频带：delta, theta, alpha, beta, gamma）
    '''
    def __init__(self, input_channels=5, hid_channels=64, output_dim=256):
        """
        Args:
            input_channels: 输入通道数，论文中应该是5（5个频带）
            hid_channels: 隐藏层通道数64
            output_dim: 输出特征维度256（与EM特征对齐）
        
        网络结构：
        1. 残差块：两个basic_block + shortcut连接
        2. CBAM注意力：通道注意力 + 空间注意力
        3. 池化层：2x2最大池化
        4. 全连接层：将特征图展平后映射到256维
        """
        super(ResCBAM, self).__init__()
        self.input_channels = input_channels
        
        # 残差块的主路径
        self.rb1 = nn.Sequential(
            basic_block(input_channels, hid_channels),   # 第一层卷积
            basic_block(hid_channels, hid_channels)      # 第二层卷积
        )
        
        # 残差连接的shortcut路径
        self.res1 = basic_block(input_channels, hid_channels)  # 1x1? 实际是3x3，保持维度一致

        # CBAM注意力模块
        self.ca1 = ChannelAttention(hid_channels)   # 通道注意力
        self.sa1 = SpatialAttention()               # 空间注意力

        self.bn1 = nn.BatchNorm2d(hid_channels)     # 批归一化
        self.maxpool = nn.MaxPool2d(2)               # 2x2最大池化，空间尺寸减半
        self.linear = nn.Linear(1024, output_dim)    # 全连接层映射到256维

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: EEG输入特征，形状 (batch_size, 5, 9, 9)
               论文中：差分熵特征，每个频带转换为9×9的2D矩阵
               5个频带：delta, theta, alpha, beta, gamma
               9×9：对应脑电通道的2D布局
        
        Returns:
            提取的EEG特征，形状 (batch_size, 256)
        """
        assert x.shape[1] == self.input_channels  # 检查输入通道数是否为5

        # ============ 残差连接 ============
        # 保存原输入，避免原地修改
        x_res = self.res1(x)      # shortcut路径: (batch, 5, 9, 9) -> (batch, 64, 9, 9)
        x_rb1 = self.rb1(x)       # 主路径: (batch, 5, 9, 9) -> (batch, 64, 9, 9)
        x_sum = x_rb1 + x_res     # 残差相加: (batch, 64, 9, 9)

        # ============ CBAM注意力 ============
        # 通道注意力：关注哪些频带/channel重要
        x_ca1 = self.ca1(x_sum) * x_sum  # 通道权重 * 特征图: (batch, 64, 9, 9)
        
        # 空间注意力：关注脑区的哪些位置重要
        x_sa1 = self.sa1(x_ca1) * x_ca1  # 空间权重 * 特征图: (batch, 64, 9, 9)

        # ============ 后处理 ============
        # 批归一化和池化
        x_bn1 = self.bn1(x_sa1)          # (batch, 64, 9, 9)
        x_pooled = self.maxpool(x_bn1)    # 2x2池化: (batch, 64, 4, 4)  [9/2=4.5 -> 4]

        # 展平并映射到256维
        x_flat = x_pooled.view(x_pooled.size(0), -1)  # (batch, 64*4*4=1024)
        output = self.linear(x_flat)                   # (batch, 256)

        return output


"""
=============================================================================
数据流形状变化追踪
=============================================================================

输入层:
    x: (batch_size, 5, 9, 9)
    │
    ├── res1: basic_block (5→64)
    │    └── x_res: (batch, 64, 9, 9)
    │
    └── rb1: [basic_block(5→64), basic_block(64→64)]
         └── x_rb1: (batch, 64, 9, 9)
          │
          + (残差连接)
          ↓
    x_sum: (batch, 64, 9, 9)
    │
    ├── ChannelAttention
    │    avg_pool: (batch, 64, 1, 1)
    │    max_pool: (batch, 64, 1, 1)
    │    fc1→relu→fc2: (batch, 64, 1, 1)
    │    ↓
    │    ca_weight: (batch, 64, 1, 1)
    │    × x_sum
    │    ↓
    │    x_ca1: (batch, 64, 9, 9)
    │
    ├── SpatialAttention
    │    avg_out: (batch, 1, 9, 9)
    │    max_out: (batch, 1, 9, 9)
    │    concat: (batch, 2, 9, 9)
    │    conv1: (batch, 1, 9, 9)
    │    ↓
    │    sa_weight: (batch, 1, 9, 9)
    │    × x_ca1
    │    ↓
    │    x_sa1: (batch, 64, 9, 9)
    │
    ├── bn1: (batch, 64, 9, 9)
    │
    ├── maxpool (2x2): (batch, 64, 4, 4)
    │
    ├── view: (batch, 64*4*4=1024)
    │
    └── linear: (batch, 256)

输出层:
    output: (batch_size, 256)

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `ResCBAM` | 2.B.1节 EEG Encoder | EEG特征提取器 |
| `rb1 + res1` | 残差块 (Residual Block) | 缓解梯度消失，加深网络 |
| `ChannelAttention` | CBAM的通道注意力 | 关注哪个频带重要 |
| `SpatialAttention` | CBAM的空间注意力 | 关注哪个脑区重要 |
| `input_channels=5` | 5个频带 | delta, theta, alpha, beta, gamma |
| 9×9输入 | 脑电通道2D布局 | 对应脑电帽的电极位置 |
| 输出256维 | 图1中的EEG特征 | 与EM特征维度对齐 |

=============================================================================
关键点解释
=============================================================================

1. **为什么EEG输入是(5, 9, 9)？**
   - 5个通道：对应5个频带(delta, theta, alpha, beta, gamma)的差分熵特征
   - 9×9：将脑电通道映射到2D网格，保持空间拓扑结构
   - 论文引用[34]的方法

2. **CBAM注意力为什么有效？**
   - 通道注意力：不同频带对情感识别的贡献不同
   - 空间注意力：不同脑区（如前额叶、颞叶）对情感处理的重要性不同
   - 两者结合：同时关注"什么频带"和"哪个脑区"

3. **为什么输出1024维然后降到256？**
   - 池化后特征图: 64通道 × 4 × 4 = 1024
   - 全连接层降维到256，与EM特征维度一致
   - 便于后续的多模态融合

4. **残差连接的作用？**
   - 解决深层网络梯度消失问题
   - 让网络可以学习恒等映射
   - 论文引用He et al.的ResNet [27]
"""