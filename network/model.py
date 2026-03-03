import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .rescnn import *
from .Cross_Att import *
from .ETnet import *


class MyModel(nn.Module):
    """
    HIA-Net整体模型 - 对应论文图1的完整框架
    
    功能：将EEG编码器、EM编码器和HAIA模块组合成完整的HIA-Net
    论文图1流程：
    EEG信号 → ResCBAM (EEG Encoder) → 
                                            → HAIA模块(多层交叉注意力+门控) → 融合特征
    EM信号  → DenseNet1D (EM Encoder)  → 
    
    输出：多层融合特征，用于后续的Few-Shot学习和MMD损失计算
    """
    def __init__(self, eeg_input_dim, eye_input_dim, output_dim):
        """
        Args:
            eeg_input_dim: EEG输入维度，实际由ResCBAM内部决定，这里可能未使用
            eye_input_dim: EM输入维度（论文中33，经过DenseNet后变成177）
            output_dim: 输出特征维度（论文中统一用256）
        
        各模块功能：
        - rescbam: EEG编码器，将(5,9,9)的EEG输入转换为256维特征
        - ETnet: EM编码器，将33维眼动特征转换为177维特征
        - encoder: HAIA模块，进行多模态融合和领域自适应
        """
        super(MyModel, self).__init__()
        
        # ============ 模态特定编码器 ============
        # EEG编码器 - 对应论文2.B.1节
        # 输入: (batch, 5, 9, 9) 的差分熵特征
        # 输出: (batch, 256) 的EEG特征向量
        self.rescbam = ResCBAM()  # ResCBAM 模块
        
        # EM编码器 - 对应论文2.B.2节
        # 输入: (batch, 33) 的眼动统计特征
        # 输出: (batch, 177) 的眼动特征向量
        self.ETnet = DenseNet1D()
        
        # ============ 多模态融合模块 ============
        # HAIA模块 - 对应论文2.C节
        # 输入: EEG特征(256) + EM特征(177)
        # 内部映射到统一维度d_model=256
        # 输出: 3层融合特征的列表，每层都是(batch, 256)
        self.encoder = MLCrossAttentionGating(
            eeg_input_dim,  # EEG维度(256)
            eye_input_dim,   # EM维度(177)
            d_model=output_dim  # 统一维度256
        )

    def forward(self, eeg_input, eye_input):
        """
        前向传播 - 对应论文图1的完整流程
        
        Args:
            eeg_input: EEG输入，形状 (batch_size, 5, 9, 9)
                     5个频带的差分熵特征，9×9的2D脑电拓扑图
            eye_input: 眼动输入，形状 (batch_size, 33)
                     33个眼动统计特征（瞳孔直径、眨眼持续时间等）
        
        Returns:
            fusion: 多层融合特征列表，每层形状 (batch_size, 256)
                   列表长度=3（对应3层DIA）
                   这些特征将用于：
                   1. 最后一层用于原型网络分类
                   2. 所有层用于HRDA模块的多层MMD损失计算
        
        数据流:
        EEG: (batch,5,9,9) ──┐
                            ├─> HAIA模块 ──> [layer1, layer2, layer3]
        EM:  (batch,33) ────┘        各层形状:(batch,256)
        """
        # 步骤1: 提取EEG特征
        # ResCBAM: (batch,5,9,9) -> (batch,256)
        eeg_features = self.rescbam(eeg_input)
        
        # 步骤2: 提取眼动特征
        # DenseNet1D: (batch,33) -> (batch,177)
        et_features = self.ETnet(eye_input)
        
        # 步骤3: 多模态融合
        # MLCrossAttentionGating: 
        #   - 输入: (batch,256)的EEG特征, (batch,177)的EM特征
        #   - 内部: EM特征先映射到256维
        #   - 输出: 3层融合特征的列表，每层(batch,256)
        fusion = self.encoder(eeg_features, et_features)
        
        return fusion  # 返回组合后的特征


"""
=============================================================================
数据流形状变化追踪
=============================================================================

完整前向传播过程:
--------------------------------------------------------------------------------
EEG分支:
    eeg_input: (batch_size, 5, 9, 9)  [5个频带, 9x9脑电拓扑图]
        ↓ ResCBAM
    eeg_features: (batch_size, 256)   [EEG特征向量]

EM分支:
    eye_input: (batch_size, 33)        [33个眼动统计特征]
        ↓ DenseNet1D
    et_features: (batch_size, 177)     [眼动特征向量]
        ↓ (在encoder内部通过linear_et映射)
    et_transformed: (batch_size, 256)  [映射到256维]

HAIA模块 (MLCrossAttentionGating):
    输入: eeg_features(256) + et_transformed(256)
    
    第1层DIA:
        cross_attention: (batch,256) -> (batch,256)
        gate: 计算通道权重
        concat+MLP: (batch,512) -> (batch,256)
        residual: (batch,256) + (batch,256) = (batch,256)
        layer1_out: (batch,256)
    
    第2层DIA:
        以layer1_out作为新的residual_output
        重复相同过程
        layer2_out: (batch,256)
    
    第3层DIA:
        重复
        layer3_out: (batch,256)
    
输出:
    fusion: [layer1_out, layer2_out, layer3_out]  # 每个都是(batch,256)

典型数值示例 (batch_size=64):
--------------------------------------------------------------------------------
- eeg_input:  (64, 5, 9, 9)
- eye_input:  (64, 33)
- eeg_features: (64, 256)
- et_features:  (64, 177)
- fusion[0]:    (64, 256)  # 第1层输出
- fusion[1]:    (64, 256)  # 第2层输出
- fusion[2]:    (64, 256)  # 第3层输出

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `MyModel` | 图1 HIA-Net整体框架 | 完整的HIA-Net模型 |
| `self.rescbam` | 2.B.1节 EEG Encoder | 提取EEG特征，带CBAM注意力 |
| `self.ETnet` | 2.B.2节 EM Encoder | 提取眼动特征，DenseNet结构 |
| `self.encoder` | 2.C节 HAIA Module | 多模态融合和领域自适应 |
| `fusion`返回值 | 2.C.2节 HRDA输入 | 多层特征用于MMD损失 |

=============================================================================
关键点解释
=============================================================================

1. **为什么返回列表而不是单个张量？**
   - 论文2.C.2节的HRDA需要计算多层MMD损失
   - 公式: L_MMD = sum(w_i * L_MMD^i) 对每一层
   - 返回列表让外层训练循环可以访问所有中间层特征

2. **维度不匹配问题？**
   - EEG输出256维，EM输出177维
   - MLCrossAttentionGating内部用linear_et将EM映射到256维
   - 确保两个模态可以在相同维度空间进行交互

3. **为什么需要两个编码器？**
   - 不同模态的数据特性不同
   - EEG: 2D拓扑结构，适合CNN处理
   - EM: 1D统计特征，适合DenseNet处理
   - 专门的编码器能更好地提取模态特定特征

4. **输入参数eeg_input_dim的作用？**
   - 代码中实际未使用，因为ResCBAM内部固定了输入通道数
   - 可能是预留接口或设计冗余
   - 实际EEG维度由ResCBAM的input_channels=5决定

=============================================================================
使用示例
=============================================================================

```python
# 初始化模型
model = MyModel(
    eeg_input_dim=256,    # 实际未使用
    eye_input_dim=177,    # EM输入维度
    output_dim=256        # 统一输出维度
)

# 前向传播
eeg_batch = torch.randn(64, 5, 9, 9)   # EEG输入
eye_batch = torch.randn(64, 33)         # 眼动输入

fusion_features = model(eeg_batch, eye_batch)  # 返回3层特征

# 使用最后一层特征进行Few-Shot分类
final_features = fusion_features[-1]  # (64, 256)

# 使用所有层特征计算多层MMD损失
for layer_features in fusion_features:
    mmd_loss = compute_mmd(layer_features)  # 对每层计算MMD
"""