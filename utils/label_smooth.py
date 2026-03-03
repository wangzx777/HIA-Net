import warnings

import torch.nn as nn
from torch.nn import functional as F
import torch


class CE_Label_Smooth_Loss(nn.Module):
    """
    标签平滑交叉熵损失 (Label Smoothing Cross Entropy Loss)
    
    功能：在标准交叉熵损失基础上，对标签进行平滑处理，防止模型过拟合
    原理：将硬标签（one-hot）转换为软标签，给错误类别分配少量概率
    
    公式：
        标准one-hot: [1, 0, 0, 0]
        标签平滑: [1-ε, ε/(C-1), ε/(C-1), ε/(C-1)]
        其中C是类别数，ε是平滑因子
    
    优点：
    1. 提高模型泛化能力
    2. 防止过拟合
    3. 对标签噪声更鲁棒
    """
    def __init__(self, classes=3, epsilon=0.15):
        """
        Args:
            classes: 类别数，论文中情感类别为3（Positive, Neutral, Negative）
            epsilon: 平滑因子，控制给错误类别的概率分配量，默认0.15
                    （给正确类别概率1-ε，每个错误类别概率ε/(C-1)）
        """
        super(CE_Label_Smooth_Loss, self).__init__()

        self.classes = classes      # 类别总数 C
        self.epsilon = epsilon       # 平滑因子 ε

    def forward(self, input, target):
        """
        计算标签平滑损失
        
        Args:
            input: 模型输出logits，形状 (batch_size, num_classes)
                   未经过softmax的原始分数
            target: 真实标签，形状 (batch_size,)
                   每个元素是0到num_classes-1的整数
        
        Returns:
            loss: 标量损失值
        
        数学推导：
            loss = -Σ [q_i * log(p_i)]
            其中：
                q_i = 1-ε  if i = target
                q_i = ε/(C-1)  otherwise
                p_i = softmax(input)_i
        """
        # 步骤1：计算log概率
        # log_softmax = log(exp(x_i)/sum(exp(x_j)))
        log_prob = F.log_softmax(input, dim=-1)  # (batch_size, num_classes)

        # 步骤2：创建平滑标签权重矩阵
        # 初始化为 ε/(C-1) 给所有类别
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.)
        # weight形状: (batch_size, num_classes)
        # 初始值: [[ε/(C-1), ε/(C-1), ε/(C-1)], ...]

        # 步骤3：将正确类别的位置设置为 1-ε
        # scatter_ 函数: 在指定维度上，根据索引填入值
        # 参数: (dim, index, value)
        # - dim=-1: 在最后一个维度（类别维度）上操作
        # - target.unsqueeze(-1): (batch_size, 1) 每个样本的真实类别索引
        # - (1. - self.epsilon): 要填入的值
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        # 最终weight: 正确类别位置为1-ε，其他位置为ε/(C-1)

        # 步骤4：计算交叉熵损失
        # -weight * log_prob: 逐元素相乘，得到每个样本每个类别的损失贡献
        # sum(dim=-1): 对类别维度求和，得到每个样本的损失
        # mean(): 对所有样本取平均
        loss = (-weight * log_prob).sum(dim=-1).mean()

        return loss


"""
=============================================================================
数据流形状变化追踪
=============================================================================

输入:
    input: (batch_size, 3)   # 例如 batch_size=64
    target: (batch_size,)
    
步骤1: log_softmax
    log_prob: (64, 3)  # log概率值
    
步骤2: 创建权重矩阵
    weight_init: (64, 3)  # 全部初始化为 ε/(3-1) = 0.15/2 = 0.075
    
步骤3: 填充正确类别
    target.unsqueeze(-1): (64, 1)  # 每个样本的真实类别
    weight_final: (64, 3)  # 正确类别位置=0.85，其他=0.075
    
步骤4: 计算损失
    (-weight * log_prob): (64, 3)  # 逐元素乘积
    sum(dim=-1): (64,)  # 每个样本的损失
    mean(): 标量  # 最终损失

数值示例 (3分类):
--------------------------------------------------------------------------------
假设:
    ε = 0.15
    C = 3
    target = [0, 2, 1]  # 第一个样本类别0，第二个类别2，第三个类别1

权重矩阵:
    样本0: [0.85, 0.075, 0.075]  # 类别0概率高
    样本1: [0.075, 0.075, 0.85]  # 类别2概率高
    样本2: [0.075, 0.85, 0.075]  # 类别1概率高

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `CE_Label_Smooth_Loss` | 可能用于2.D节损失函数 | 替代标准交叉熵损失 |
| `classes=3` | 表I | 3种情感类别 |
| `epsilon=0.15` | 超参数 | 平滑因子，控制正则化强度 |

=============================================================================
标签平滑 vs 标准交叉熵
=============================================================================

标准交叉熵:
    target = [1, 0, 0]  # one-hot硬标签
    鼓励模型输出正确类别的概率为1，其他为0
    可能导致过拟合，模型过于自信

标签平滑:
    target = [0.85, 0.075, 0.075]  # 软标签
    允许模型犯一些小错误，不那么绝对
    正则化效果，提高泛化能力

损失对比示例:
--------------------------------------------------------------------------------
假设模型预测概率: [0.9, 0.05, 0.05]  # 很自信

标准交叉熵:
    loss = -log(0.9) = 0.105

标签平滑:
    loss = -(0.85*log(0.9) + 0.075*log(0.05) + 0.075*log(0.05))
         = -(0.85*(-0.105) + 0.075*(-2.996) + 0.075*(-2.996))
         = -(-0.089 - 0.225 - 0.225)
         = 0.539

标签平滑的损失更大，因为模型过于自信被惩罚

=============================================================================
关键点解释
=============================================================================

1. **为什么需要标签平滑？**
   - 标准交叉熵鼓励模型输出one-hot分布，可能过拟合
   - 真实数据往往有噪声，标签可能不完美
   - 平滑后的标签让模型学习更鲁棒的特征

2. **epsilon参数的作用？**
   - 控制正则化的强度
   - epsilon越大，平滑程度越高
   - 通常设置在0.1-0.2之间

3. **scatter_函数的作用？**
   - 原地操作，根据索引填充值
   - 用于构建平滑标签矩阵
   - 比循环更高效

4. **什么时候使用标签平滑？**
   - 数据集有标签噪声时
   - 模型过拟合时
   - 想要更鲁棒的模型时

=============================================================================
使用示例
=============================================================================

```python
# 初始化损失函数
criterion = CE_Label_Smooth_Loss(classes=3, epsilon=0.15)

# 在训练循环中使用
for epoch in range(n_epochs):
    for batch in dataloader:
        eeg, eye, labels = batch
        
        # 前向传播
        features = model(eeg, eye)
        final_features = features[-1]  # 取最后一层特征
        logits = classifier(final_features)  # (batch_size, 3)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""