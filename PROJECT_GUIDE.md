# HIA-Net 项目说明指南

## 目录
1. [项目概述](#项目概述)
2. [项目结构](#项目结构)
3. [文件详细说明](#文件详细说明)
4. [与论文对应关系](#与论文对应关系)
5. [核心架构流程](#核心架构流程)
6. [数据流说明](#数据流说明)
7. [关键类和函数说明](#关键类和函数说明)

---

## 项目概述

**HIA-Net (Hierarchical Interactive Alignment Network)** 是一个用于多模态小样本情感识别的深度学习框架。

### 主要特点：
- **多模态融合**：结合 EEG（脑电图）和眼动追踪数据
- **小样本学习**：采用原型网络（Prototypical Networks）进行少样本分类
- **域适应**：使用 GDD（Geodesic Distance Discrepancy）进行跨被试域适应
- **层次化交互**：通过多层交叉注意力机制实现模态间信息交互

### 支持的数据集：
- **SEED**：上海交大情感识别数据集（15被试，3 session）
- **SEED-Franch**：扩展版（8被试，3 session）

---

## 项目结构

```
HIA-Net/
├── train.py                          # 主训练脚本（SEED数据集）
├── train_franch.py                   # 主训练脚本（SEED-Franch数据集）
├── pyproject.toml                    # 项目配置
├── README.md                         # 项目说明
├── .gitignore                        # Git忽略配置
│
├── network/                          # 网络模型定义
│   ├── model.py                      # 主模型MyModel
│   ├── Cross_Att.py                  # 交叉注意力模块
│   ├── rescnn.py                     # ResCBAM特征提取器
│   ├── proto_att.py                  # 原型网络
│   └── ETnet.py                      # 眼动DenseNet
│
├── data_prepare/                     # 数据准备
│   ├── load_data.py                  # 数据加载和处理
│   └── prototypical_batch_sampler.py # 原型批次采样器
│
├── utils/                            # 工具函数
│   ├── prototypical_loss.py          # 原型损失函数
│   ├── gdd.py                        # GDD域适应损失
│   ├── early_stop.py                 # 早停机制
│   ├── utils.py                      # 通用工具
│   └── label_smooth.py               # 标签平滑
│
└── parse/                            # 参数解析
    └── parser_camp.py                # 命令行参数定义
```

---

## 文件详细说明

### 1. 训练脚本

#### `train.py`
**作用**：SEED数据集的主训练脚本

**核心功能**：
- 实现leave-one-subject-out交叉验证
- 3个session循环，每session 12被试
- 每个被试作为目标域，其余作为源域
- 包含训练、验证、测试完整流程
- 早停机制防止过拟合
- TensorBoard日志记录

**关键参数**：
```python
eeg_input_dim = 256        # EEG特征维度
eye_input_dim = 177        # 眼动特征维度
output_dim = 256           # 融合输出维度
emotion_categories = 3     # 情感类别数（正/中/负）
```

**与论文对应**：
- Section IV-A: Experimental Setup
- Section IV-C: Implementation Details

---

#### `train_franch.py`
**作用**：SEED-Franch数据集的训练脚本

**与train.py区别**：
- 使用8被试而非12被试
- 文件名列表不同（包含10-14号被试）

---

### 2. 网络模型 (`network/`)

#### `model.py`
**作用**：主模型定义

**类**：`MyModel`

**结构**：
```
输入: EEG (batch, 5, 9, 9) + Eye (batch, 177)
    ↓
ResCBAM (EEG特征提取) → 256-dim
DenseNet1D (Eye特征提取) → 256-dim
    ↓
MLCrossAttentionGating (融合) → 5层输出
    ↓
输出: 多层级融合特征
```

**与论文对应**：
- Section III-A: Hierarchical Interactive Alignment Network
- Fig. 2: Overall architecture of HIA-Net

---

#### `Cross_Att.py`
**作用**：交叉注意力模块实现

**类**：
1. `CrossAttentionLayer` - 单层交叉注意力
2. `MLCrossAttentionGating` - 多层带门控的交叉注意力

**MLCrossAttentionGating结构**：
```python
输入: EEG特征 (256-dim) + Eye特征 (256-dim)
    ↓
线性变换: EEG → EEG, Eye → 256-dim
    ↓
3层Cross-Attention:
    - 每层: Eye作为Query，EEG作为Key/Value
    - 门控机制: Sigmoid(Gate) 加权
    - 残差连接
    - MLP融合
    ↓
输出: 每层中间结果 (共5层，用于GDD多层级对齐)
```

**与论文对应**：
- Section III-A: "Hierarchical Cross-Modal Attention Module"
- Fig. 2(b): The detailed structure of HCM
- Equation (3)-(6): Attention and gating mechanism

---

#### `rescnn.py`
**作用**：EEG特征提取器 (ResCBAM)

**类**：`ResCBAM`

**结构**：
```python
输入: (batch, 5, 9, 9) - 5频段EEG，9×9空间布局
    ↓
Basic Block 1:
    - Conv(5→64) + BN + ReLU
    - Conv(64→64) + BN + ReLU
    ↓
残差连接: Input → Conv(5→64)
    ↓
注意力:
    - ChannelAttention(64) - 通道注意力
    - SpatialAttention() - 空间注意力
    ↓
BN + MaxPool(2×2)
    ↓
Flatten → Linear(1024 → 256)
    ↓
输出: 256-dim EEG特征
```

**与论文对应**：
- Section III-A: "Convolutional CBAM for Spatial-Spectral Feature Extraction"
- CBAM from Woo et al. "CBAM: Convolutional Block Attention Module"

---

#### `ETnet.py`
**作用**：眼动特征提取器

**类**：`DenseNet1D`

**结构**：
```python
输入: (batch, 177) - 眼动特征向量
    ↓
Unsqueeze → (batch, 177, 1)
    ↓
Dense Block 1:
    - Layer1: BN→ReLU→Conv1d(177→12) → (batch, 12, 1)
    - Layer2: BN→ReLU→Conv1d(189→12) [concat layer1] → (batch, 12, 1)
    - ... (4 layers)
    ↓
Conv1d(237→237) - 特征压缩
    ↓
Dense Block 2: (4 layers)
    ↓
Conv1d
    ↓
Dense Block 3: (4 layers)
    ↓
Conv1d
    ↓
Global Average Pooling
    ↓
输出: 256-dim Eye特征
```

**与论文对应**：
- Section III-A: "DenseNet for Eye Movement Feature Extraction"

---

#### `proto_att.py`
**作用**：原型网络实现

**类**：`ProtoNet`

**功能**：
```python
输入: 融合特征 (support + query samples)
    ↓
分support和query:
    - support: n_classes × n_support 样本
    - query: n_classes × n_query 样本
    ↓
计算原型 (Prototypes):
    - 对每个类别，support样本取平均
    - prototype_c = mean(support_samples_of_class_c)
    ↓
计算距离:
    - 每个query样本到所有prototypes的欧氏距离
    - dists = euclidean_dist(query, prototypes)
    ↓
分类:
    - 最小距离对应的类别为预测结果
    - 负距离经过softmax得到分类概率
```

**与论文对应**：
- Section III-B: "Prototypical Few-Shot Emotion Recognition"
- Equation (7)-(9): Prototype computation and classification
- Snell et al. "Prototypical Networks for Few-shot Learning"

---

### 3. 数据准备 (`data_prepare/`)

#### `load_data.py`
**作用**：数据加载和预处理

**主要函数**：
1. `load_data()`: 从文件加载EEG和眼动数据
2. `load4data()`: 将数据转换为Tensor
3. `manual_split()`: 手动划分support/query集
4. `convert_chl()`: 1D通道转换为2D拓扑图
5. `standardize_bands()`: 频段标准化
6. `exact_bands()`: 提取DE特征频段

**数据预处理流程**：
```python
原始数据 (.npz EEG, .pkl Eye)
    ↓
加载EEG: pickle loads → 5频段DE特征 (delta, theta, alpha, beta, gamma)
    ↓
标准化: StandardScaler每个频段
    ↓
提取频段: stack成 (samples, 62, 5)
    ↓
1D→2D转换: convert_chl() → (samples, 5, 9, 9)  # 62通道映射到9×9拓扑
    ↓
加载Eye: pickle loads → 原始特征
    ↓
归一化: MinMaxScaler → (samples, 177)
    ↓
输出: EEG (samples, 5, 9, 9), Eye (samples, 177), Labels (samples,)
```

**与论文对应**：
- Section IV-A: "Dataset and Preprocessing"
- 62通道到9×9网格的映射基于EEG电极的拓扑位置

---

#### `prototypical_batch_sampler.py`
**作用**：小批次原型采样器

**类**：`PrototypicalBatchSampler`

**功能**：
```python
输入: 所有样本的标签
    ↓
每次迭代:
    1. 随机选择N个类别 (classes_per_it)
    2. 对每个类别:
        - 随机选择K个样本 (sample_per_class)
    3. 将这N×K个样本的索引返回
    ↓
输出: 用于当前episode的样本索引
```

**作用**：确保每个training episode（迭代）中，采样的样本类别平衡，适合小样本原型学习。

**与论文对应**：
- Section IV-A: "Training Strategy"
- N-way K-shot采样策略

---

### 4. 工具函数 (`utils/`)

#### `prototypical_loss.py`
**作用**：原型损失函数实现

**主要函数**：
1. `prototypical_loss()`: 基础原型损失（单个域）
2. `prototypical_loss2()`: 扩展版本（支持混淆矩阵）
3. `prototypical_loss_aggregate()`: 多源域蒸馏版本

**计算流程**：
```python
输入: 距离矩阵dists (query到各prototype的距离)
    ↓
负距离softmax: log_p_y = log_softmax(-dists)
    ↓
计算负对数似然损失:
    - 对每个query样本，取真实类别的log概率
    - 取平均得到最终损失
    ↓
计算准确率:
    - 取概率最大的类别作为预测
    - 与真实标签比较
```

**与论文对应**：
- Section III-B: "Prototypical Few-Shot Emotion Recognition"
- Equation (8)-(9): Loss computation

---

#### `gdd.py`
**作用**：Geodesic Distance Discrepancy (GDD) 域适应损失

**主要函数**：
1. `guassian_kernel()`: 计算高斯核矩阵
2. `gdd()`: 计算GDD损失

**数学原理**：
```python
GDD通过计算源域和目标域在RKHS (再生核希尔伯特空间) 中的距离来实现域适应

输入: 源域特征X_s, 目标域特征X_t
    ↓
计算核矩阵K:
    K = [K_ss  K_st
         K_ts  K_tt]
    ↓
计算GDD距离:
    GDD = ||μ_s - μ_t||^2_H
        = E[K_ss] - 2*E[K_st] + E[K_tt]
    ↓
输出: 域适应损失
```

**与论文对应**：
- Section III-C: "Multi-Level Interactive Alignment"
- Equation (10)-(12): GDD loss formulation
- Multi-level: 在融合网络的5个层级分别计算GDD

---

#### `early_stop.py`
**作用**：早停机制

**类**：`EarlyStoppingAccuracy`

**功能**：
- 监控验证集准确率
- 当验证准确率不再提升时，保存最佳模型
- 当连续多个epoch没有提升时，触发早停
- 同时保存last_model（最后epoch的模型）和best_model（最佳模型）

---

### 5. 参数解析 (`parse/`)

#### `parser_camp.py`
**作用**：命令行参数定义

**主要参数类别**：

1. **训练参数**：
   - `epochs`: 训练轮数 (默认50)
   - `learning_rate`: 学习率 (默认1e-4)
   - `patience`: 早停耐心值 (默认15)
   - `iterations`: 每轮迭代次数 (默认20)

2. **小样本参数**：
   - `classes_per_it_src/tgt`: 每轮类别数 (默认3)
   - `num_support_src/tgt`: 支持集样本数 (默认1)
   - `num_query_src/tgt`: 查询集样本数 (默认20)

3. **其他**：
   - `cuda`: GPU设备ID
   - `seed`: 随机种子
   - `num_bands`: EEG频段数 (SEED为5)

---

## 与论文对应关系

### 论文结构回顾

论文《HIA-Net: Hierarchical Interactive Alignment Network for Multimodal Few-Shot Emotion Recognition》主要结构：

- **Section I**: Introduction
- **Section II**: Related Work
- **Section III**: Proposed Method
  - A. Hierarchical Interactive Alignment Network
  - B. Prototypical Few-Shot Emotion Recognition
  - C. Multi-Level Interactive Alignment
- **Section IV**: Experiments
- **Section V**: Conclusion

### 详细对应关系

| 论文部分 | 对应代码文件 | 说明 |
|---------|-------------|------|
| **Section III-A**<br>Hierarchical Interactive<br>Alignment Network | `network/model.py`<br>`network/rescnn.py`<br>`network/ETnet.py`<br>`network/Cross_Att.py` | 整体架构实现：<br>- ResCBAM: EEG特征提取<br>- DenseNet1D: 眼动特征提取<br>- MLCrossAttentionGating: 多层次交叉注意力融合<br>- 返回5层中间输出用于多层次对齐 |
| **Section III-B**<br>Prototypical Few-Shot<br>Emotion Recognition | `network/proto_att.py`<br>`utils/prototypical_loss.py` | 原型网络实现：<br>- 原型计算（类内平均）<br>- 欧氏距离计算<br>- 基于距离的softmax分类<br>- 损失函数和准确率计算 |
| **Section III-C**<br>Multi-Level<br>Interactive Alignment | `utils/gdd.py`<br>`train.py` (lines 200-226) | 多层次交互对齐：<br>- GDD损失计算（高斯核+域距离）<br>- 对5个融合层分别计算GDD<br>- 对数权重加权（公式中的a*log(bi+1)）<br>- 动态gamma权重（2/(1+exp(-10p))-1） |
| **Section IV-A**<br>Dataset and<br>Preprocessing | `data_prepare/load_data.py` | 数据预处理：<br>- EEG 62通道DE特征加载<br>- 9×9拓扑图转换（convert_chl）<br>- 5频段标准化（standardize_bands）<br>- 眼动特征MinMax归一化<br>- 支持/查询集手动划分 |
| **Section IV-A**<br>Training Strategy | `data_prepare/prototypical_batch_sampler.py`<br>`parse/parser_camp.py` | 训练策略：<br>- N-way K-shot采样（3-way 1-shot默认）<br>- 每轮20个episode<br>- 早停机制（patience=15）<br>- Adam优化器（lr=1e-4） |

---

## 核心架构流程

### 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           输入数据                                       │
│  EEG: (batch, 5, 9, 9)          Eye: (batch, 177)                      │
│  [5频段×9×9拓扑图]               [177维眼动特征]                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      特征提取 (Feature Extraction)                       │
│                                                                         │
│   EEG Branch (ResCBAM)              Eye Branch (DenseNet1D)             │
│   ┌─────────────────────┐           ┌─────────────────────┐             │
│   │ ResBlock + CBAM     │           │ Dense Block ×3      │             │
│   │ - Conv + BN + ReLU  │           │ - BN + ReLU + Conv  │             │
│   │ - Channel Attention │           │ - Growth Rate: 12   │             │
│   │ - Spatial Attention  │           │ - 4 layers/block    │             │
│   └─────────────────────┘           └─────────────────────┘             │
│            ↓                                    ↓                       │
│      EEG features (256-dim)           Eye features (256-dim)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    交叉模态融合 (Cross-Modal Fusion)                      │
│              MLCrossAttentionGating - 3层结构                           │
│                                                                         │
│  输入: EEG_features (256) + ET_features (256)                           │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Layer 1:                                                    │       │
│  │   - Eye作为Query，EEG作为Key/Value                          │       │
│  │   - CrossAttention: Q×K^T/sqrt(d) → Softmax → ×V          │       │
│  │   - Gate: σ(W·Attn) 加权                                    │       │
│  │   - Concat + MLP融合                                        │       │
│  │   - Residual: EEG_input + Fused                             │       │
│  │   → Output_1 (保存用于GDD)                                   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Layer 2: (同样结构，输入是Layer 1的输出)                      │       │
│  │   → Output_2 (保存用于GDD)                                   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Layer 3: (同样结构)                                          │       │
│  │   → Output_3 (保存用于GDD)                                   │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                              ↓                                          │
│  注意：实际代码返回5层输出（3层 + 2层残差）用于多层级GDD对齐              │
│                                                                         │
│  输出: [out1, out2, out3, out4, out5] - 每层都是256-dim                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    原型分类 (Prototypical Classification)                 │
│                                                                         │
│  输入: 融合特征 (source + target)                                        │
│                              ↓                                          │
│  分Source和Target:                                                       │
│    - Source: 用于计算原型 (prototypes)                                   │
│    - Target: 用于查询 (queries)                                          │
│                              ↓                                          │
│  计算原型:                                                               │
│    - 对每个类别c: prototype_c = mean(support_samples_class_c)            │
│                              ↓                                          │
│  计算欧氏距离:                                                            │
│    - dist(query, prototype) = ||query - prototype||^2                   │
│                              ↓                                          │
│  Softmax分类:                                                            │
│    - log_prob = log_softmax(-distances)                                  │
│    - 预测类别 = argmax(log_prob)                                         │
│                              ↓                                          │
│  损失和准确率:                                                            │
│    - loss = -log_prob[true_class].mean()                                 │
│    - accuracy = (pred == true).mean()                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    域适应损失 (Domain Adaptation - GDD)                   │
│                                                                         │
│  输入: Source和Target在融合网络各层的输出                                  │
│         [src_out1, src_out2, ..., src_out5]                             │
│         [tgt_out1, tgt_out2, ..., tgt_out5]                             │
│                              ↓                                          │
│  对每一层l计算GDD:                                                        │
│    GDD_l = ||μ_s^l - μ_t^l||_H^2  (RKHS空间中的距离)                      │
│                              ↓                                          │
│  多核高斯核计算:                                                          │
│    K(x,y) = Σ_i exp(-||x-y||^2 / σ_i)                                    │
│    σ_i = bandwidth × (kernel_mul)^i                                     │
│                              ↓                                          │
│  GDD计算:                                                                │
│    XX = K_ss的平均, YY = K_tt的平均                                       │
│    XY = K_st的平均, YX = K_ts的平均                                       │
│    GDD = (XX + XY).sum() + (YX + YY).sum()                               │
│                              ↓                                          │
│  多层级加权:                                                             │
│    - 权重: w_i = a × log(b×i + 1), 然后归一化                              │
│    - 默认: a=0.5, b=1                                                    │
│    - GDD_total = Σ_i w_i × GDD_i                                         │
│                              ↓                                          │
│  动态gamma:                                                              │
│    - γ = 2/(1+exp(-10×p)) - 1  # p为当前epoch比例                        │
│    - 用于逐渐增加GDD损失的权重                                            │
│    ↓                                                                     │
│  总损失:                                                                  │
│    Loss = Proto_loss + γ × GDD_total                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 数据流说明

### 训练时的数据流

```
Epoch Loop (50 epochs)
    ↓
Session Loop (3 sessions)
    ↓
Subject Loop (12 subjects for SEED)
    ↓
    Source Subjects: 11 others
    Target Subject: current
    ↓
Data Loading:
    EEG: (samples, 5, 9, 9)
    Eye: (samples, 177)
    Labels: (samples,)
    ↓
Episode Loop (20 iterations per epoch)
    ↓
    Sample Batches (PrototypicalBatchSampler):
        - Source: 3 classes × (1 support + 20 query) = 63 samples
        - Target: 3 classes × (1 support + 20 query) = 63 samples
    ↓
    Forward Pass:
        EEG → ResCBAM → 256-dim
        Eye → DenseNet1D → 256-dim
        Fusion → MLCrossAttentionGating → 5 outputs
    ↓
    ProtoNet:
        Source fusion → prototypes
        Target fusion → queries
        Compute distances → log_probs
    ↓
    Loss Computation:
        Proto_loss = CE_loss(log_probs, labels)
        GDD_loss = Σ w_i × GDD(layer_i_source, layer_i_target)
        Total_loss = Proto_loss + γ × GDD_loss
    ↓
    Backward Pass → Update Weights
    ↓
Validation (after each epoch)
    ↓
Early Stopping Check
    ↓
Test (after training ends)
    Save best and last model results
```

---

## 关键类和函数说明

### 类说明

#### 1. `MyModel` (network/model.py)
```python
class MyModel(nn.Module):
    def __init__(self, eeg_input_dim, eye_input_dim, output_dim):
        self.rescbam = ResCBAM()  # EEG特征提取
        self.ETnet = DenseNet1D()  # 眼动特征提取
        self.encoder = MLCrossAttentionGating(...)  # 融合编码器
```

#### 2. `MLCrossAttentionGating` (network/Cross_Att.py)
```python
class MLCrossAttentionGating(nn.Module):
    def __init__(self, eeg_dim, et_dim, d_model, num_layers=3):
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model) for _ in range(num_layers)
        ])
        self.gate = nn.Linear(d_model, d_model)  # 门控
        self.fusion_mlp = nn.Sequential(...)  # 融合MLP
```

#### 3. `ProtoNet` (network/proto_att.py)
```python
class ProtoNet(nn.Module):
    def forward(self, x, y, n_classes, n_support, n_query, flag):
        # flag=0: support和query来自同一域
        # flag=1: support来自source，query来自target
        # 返回: query到各prototype的距离
```

### 函数说明

#### 1. `gdd()` (utils/gdd.py)
```python
def gdd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域和目标域之间的GDD距离

    Args:
        source: 源域特征 (n_samples, n_features)
        target: 目标域特征 (m_samples, n_features)
        kernel_mul: 核带宽乘数
        kernel_num: 多核数量

    Returns:
        loss: GDD距离（标量）
    """
```

#### 2. `prototypical_loss2()` (utils/prototypical_loss.py)
```python
def prototypical_loss2(dists, n_classes, n_query, opt):
    """
    计算原型网络的损失和准确率

    Args:
        dists: query到各prototype的距离矩阵
        n_classes: 类别数
        n_query: 每类的query样本数
        opt: 配置选项

    Returns:
        loss_val: 损失值
        acc_val: 准确率
        conf_matrix: 混淆矩阵
    """
```

#### 3. `load4data()` (data_prepare/load_data.py)
```python
def load4data(parser, eeg_path, eye_path, eeg_session_names, eye_session_names, sess_idx, idx, mode):
    """
    加载SEED数据集用于迁移学习

    Args:
        parser: 参数解析器
        eeg_path: EEG数据路径
        eye_path: 眼动数据路径
        eeg_session_names: EEG文件名列表
        eye_session_names: 眼动文件名列表
        sess_idx: session索引
        idx: 被试索引列表
        mode: 模式 ('full'/'train'/'val'/'test')

    Returns:
        eeg_sample: EEG样本张量
        eye_sample: 眼动样本张量
        labels: 标签张量
    """
```

---

## 总结

这份代码完整实现了论文《HIA-Net: Hierarchical Interactive Alignment Network for Multimodal Few-Shot Emotion Recognition》中提出的方法，包括：

1. **层次化交互对齐网络**（Hierarchical Interactive Alignment）
   - 基于CBAM的EEG特征提取（ResCBAM）
   - 基于DenseNet的眼动特征提取（DenseNet1D）
   - 多层交叉注意力融合（MLCrossAttentionGating）

2. **原型小样本情感识别**（Prototypical Few-Shot）
   - 原型计算（Prototype Computation）
   - 基于距离的分类（Distance-based Classification）
   - 原型损失（Prototypical Loss）

3. **多层次交互对齐**（Multi-Level Interactive Alignment）
   - 多层级GDD损失（Multi-level GDD Loss）
   - 对数权重加权（Logarithmic Weighting）
   - 动态Gamma调整（Dynamic Gamma Annealing）

代码结构清晰，模块分工明确，完整复现了论文的所有关键组件。