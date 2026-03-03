import torch
import torch.nn as nn


class DenseBlock1D(nn.Module):
    """
    DenseNet的一维密集连接块 - 对应论文中DenseNet的基础构建块
    
    功能：实现密集连接机制，每一层的输入是前面所有层输出的拼接
    特点：缓解梯度消失，促进特征复用
    """
    def __init__(self, input_channels, growth_rate, num_layers):
        """
        Args:
            input_channels: 输入特征图的通道数
            growth_rate: 增长率，每层新产生的特征图数量（论文中设为12）
            num_layers: 当前密集块中包含的卷积层数（论文中每块4层）
        
        数据形状说明：
        - 输入: (batch_size, channels, length)
        - channels维度会逐层增长，length维度保持不变
        """
        super(DenseBlock1D, self).__init__()
        self.layers = nn.ModuleList()
        
        # 创建密集块内的所有层
        # 第i层的输入通道数 = 初始通道数 + i * growth_rate
        # 因为前面i层每层都产生了growth_rate个新特征图
        for i in range(num_layers):
            # 计算当前层的输入通道数
            in_ch = input_channels + i * growth_rate  # 随着层数增加而增加
            self.layers.append(self._make_layer(in_ch, growth_rate))

    def _make_layer(self, in_channels, out_channels):
        """
        创建密集块中的单个层：BN + ReLU + Conv1d
        
        Args:
            in_channels: 输入通道数（随着层数增加而增加）
            out_channels: 输出通道数（固定为growth_rate）
        
        结构：BN -> ReLU -> Conv1d (kernel_size=3, padding=1)
        - kernel_size=3：捕捉局部时间依赖性
        - padding=1：保持序列长度不变
        """
        layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),  # 批归一化，加速训练
            nn.ReLU(),                     # 激活函数，引入非线性
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)  # 1D卷积
        )
        return layer

    def forward(self, x):
        """
        前向传播 - 实现密集连接
        
        Args:
            x: 输入特征，形状 (batch_size, channels, length)
        
        Returns:
            拼接后的特征，形状 (batch_size, channels + num_layers*growth_rate, length)
        
        过程说明：
        1. 初始features列表包含原始输入x
        2. 对每一层：
           - 将所有之前特征在通道维拼接
           - 输入当前层生成新特征
           - 将新特征加入列表
        3. 最后将所有特征在通道维拼接
        """
        features = [x]  # 保存所有层输出的列表，初始包含原始输入
        
        for layer in self.layers:
            # 将之前所有层的输出在通道维拼接 (dim=1)
            # 形状: (batch_size, in_channels, length)
            concat_features = torch.cat(features, dim=1)
            
            # 通过当前层生成新特征
            # 输出形状: (batch_size, growth_rate, length)
            new_feature = layer(concat_features)
            
            # 将新特征加入列表，供后续层使用
            features.append(new_feature)
        
        # 将所有特征在通道维拼接
        # 最终通道数 = 输入通道数 + num_layers * growth_rate
        # 形状: (batch_size, input_channels + num_layers*growth_rate, length)
        return torch.cat(features, dim=1)


class DenseNet1D(nn.Module):
    """
    一维DenseNet - 对应论文2.B.2节的EM Encoder
    
    功能：从眼动信号中提取层次化特征
    论文原文："DenseNet for EM feature extraction. Its densely connected architecture 
              facilitates feature transfer and reuse, captures hierarchical characteristics 
              in eye movement data, alleviates vanishing gradients."
    
    论文中的EM特征：
    - 输入：33个眼动特征（瞳孔直径、眨眼持续时间等统计特征）
    - 经过DenseNet提取后得到256维的特征向量
    """
    def __init__(self, input_channels=33, growth_rate=12, block_layers=[4,4,4]):
        """
        Args:
            input_channels: 输入特征维度（论文中EM提取33个特征）
            growth_rate: 增长率，每层新增12个特征图
            block_layers: 每个密集块中的层数，[4,4,4]表示3个密集块，每块4层
        
        网络结构：
        输入(33) -> DenseBlock1(33+4*12=81) -> Conv1x1(81) -> 
        DenseBlock2(81+4*12=129) -> Conv1x1(129) -> 
        DenseBlock3(129+4*12=177) -> Conv1x1(177) -> GlobalAvgPool -> 输出(177)
        
        注：最终输出177维，再通过论文中的linear_et映射到256维
        """
        super(DenseNet1D, self).__init__()
        self.dense_blocks = nn.ModuleList()
        num_channels = input_channels  # 当前通道数，从33开始
        
        # 构建多个密集块
        for num_layers in block_layers:
            # 1. 添加密集块：特征图数量增加 num_layers * growth_rate
            self.dense_blocks.append(
                DenseBlock1D(num_channels, growth_rate, num_layers)
            )
            num_channels += num_layers * growth_rate  # 更新通道数
            
            # 2. 添加1x1卷积（过渡层）：压缩特征，保持通道数不变
            # 论文中的DenseNet使用1x1卷积来减少通道数，这里保持相同
            self.dense_blocks.append(
                nn.Conv1d(num_channels, num_channels, kernel_size=1)
            )
        
        # 全局平均池化：将变长的序列长度压缩为1
        # 输入形状: (batch_size, channels, length)
        # 输出形状: (batch_size, channels, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 眼动特征输入，形状 (batch_size, 33)
              论文中EM的33个特征包括：
              - 瞳孔直径
              - 眨眼持续时间
              - 注视点数量
              - 平均注视时间
              - 等统计特征
        
        Returns:
            提取的眼动特征，形状 (batch_size, 177)
            注意：最终输出的177维会通过MLCrossAttentionGating中的
                 linear_et映射到256维与EEG特征对齐
        
        数据处理流程：
        1. unsqueeze(2): 添加序列维度，将1D向量变为适合Conv1d处理的格式
           (batch_size, 33) -> (batch_size, 33, 1)
        
        2. 通过密集块序列：
           - 每个密集块增加通道数，保持长度不变
           - 1x1卷积压缩特征
        
        3. 全局平均池化：将长度维度(1)压缩掉
           (batch_size, 177, 1) -> (batch_size, 177)
        """
        # 添加序列维度，因为Conv1d需要 (batch, channels, length) 格式
        x = x.unsqueeze(2)  # (batch_size, 33) -> (batch_size, 33, 1)
        
        # 依次通过所有密集块和过渡层
        for block in self.dense_blocks:
            x = block(x)  # 每步都保持 (batch_size, channels, 1) 的格式
        
        # 全局平均池化 + 压缩序列维度
        x = self.global_pool(x).squeeze(-1)  # (batch_size, 177, 1) -> (batch_size, 177)
        
        return x
    
'''
输入层: (batch_size, 33)
    ↓ unsqueeze(2)
Stage 0: (batch_size, 33, 1)
    ↓ DenseBlock1 (4层)
Stage 1: (batch_size, 33+4*12=81, 1)  # 通道数增加
    ↓ Conv1x1
Stage 2: (batch_size, 81, 1)           # 通道数不变
    ↓ DenseBlock2 (4层)
Stage 3: (batch_size, 81+4*12=129, 1)  # 通道数增加
    ↓ Conv1x1
Stage 4: (batch_size, 129, 1)          # 通道数不变
    ↓ DenseBlock3 (4层)
Stage 5: (batch_size, 129+4*12=177, 1) # 最终通道数177
    ↓ GlobalAvgPool
输出:   (batch_size, 177)
'''