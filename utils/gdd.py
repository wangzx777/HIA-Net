import numpy as np
import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算高斯核矩阵 - 对应论文公式(6)中的核函数
    
    功能：计算源域和目标域样本之间的相似度矩阵，用于MMD距离计算
    
    Args:
        source: 源域特征，形状 (n_samples_src, feature_size)
        target: 目标域特征，形状 (n_samples_tgt, feature_size)
        kernel_mul: 核函数带宽的乘数因子，用于生成多尺度核
        kernel_num: 多核数量（论文中可能使用多核MMD）
        fix_sigma: 是否固定带宽，None表示自适应计算
    
    Returns:
        kernel_val: 核矩阵，形状 (n_samples_src + n_samples_tgt, n_samples_src + n_samples_tgt)
                   矩阵结构：[K_ss K_st; K_ts K_tt]
    
    数学原理：
        高斯核函数：k(x,y) = exp(-||x-y||²/(2σ²))
        多核MMD：使用多个不同带宽σ的核函数，求和得到更鲁棒的距离度量
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    
    # 步骤1：合并源域和目标域样本
    # 例如：source有10个样本，target有10个样本，total形状为(20, feature_size)
    total = torch.cat([source, target], dim=0)  # 合并在一起

    # 步骤2：计算所有样本对之间的L2距离
    # total0: 将total复制n_samples行，每行都是完整的total
    # total1: 将total转置复制，形成列向量
    # 两者相减得到所有样本对的差值
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    
    # 计算L2距离 ||x-y||²
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 形状 (n_samples, n_samples)

    # 步骤3：计算高斯核的带宽参数 σ (bandwidth)
    if fix_sigma:
        # 使用固定的带宽
        bandwidth = fix_sigma
    else:
        # 自适应计算带宽：取所有距离的均值
        # 除以(n_samples² - n_samples)是为了去除对角线上的0
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    # 步骤4：生成多尺度核的带宽列表
    # 例如：如果kernel_num=5, kernel_mul=2.0
    # 带宽列表为 [bandwidth/4, bandwidth/2, bandwidth, bandwidth*2, bandwidth*4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 步骤5：计算多个高斯核并求和
    # 对每个带宽计算高斯核：exp(-||x-y||² / bandwidth)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    # 返回多核的总和
    return sum(kernel_val)  # 将多个核合并在一起


def gdd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    GDD (Gaussian Distance Discrepancy) - MMD损失函数
    对应论文公式(6)中的L_MMD
    
    Args:
        source: 源域特征，形状 (n_samples_src, feature_size)
        target: 目标域特征，形状 (n_samples_tgt, feature_size)
        kernel_mul: 核函数乘子
        kernel_num: 核函数数量
        fix_sigma: 固定带宽
    
    Returns:
        loss: MMD距离标量值
    
    数学公式：
        MMD² = || 1/n_s * sum(φ(x_s)) - 1/n_t * sum(φ(x_t)) ||²_H
             = 1/n_s² * sum(k(x_s,x_s')) + 1/n_t² * sum(k(x_t,x_t')) 
               - 2/(n_s*n_t) * sum(k(x_s,x_t))
    
    其中k是核函数（这里使用多尺度高斯核）
    """
    n = int(source.size()[0])  # 源域样本数
    m = int(target.size()[0])  # 目标域样本数

    # 步骤1：计算合并的核矩阵
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    # 步骤2：分割核矩阵为四个子矩阵
    # kernels矩阵结构：
    # [ K_ss  K_st ]  source部分在前n行，target部分在后m行
    # [ K_ts  K_tt ]  
    XX = kernels[:n, :n]  # K_ss: 源域内部相似度 (n, n)
    YY = kernels[n:, n:]  # K_tt: 目标域内部相似度 (m, m)
    XY = kernels[:n, n:]  # K_st: 源域与目标域相似度 (n, m)
    YX = kernels[n:, :n]  # K_ts: 目标域与源域相似度 (m, n)

    # 步骤3：计算MMD的各个分量
    # XX项：1/n² * sum(k(x_s, x_s'))
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    
    # XY项：-2/(n*m) * sum(k(x_s, x_t)) 的一半
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    # YX项：-2/(n*m) * sum(k(x_t, x_s)) 的另一半（与XY对称）
    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    
    # YY项：1/m² * sum(k(x_t, x_t'))
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target
    
    # 步骤4：组合所有项得到MMD距离
    # loss = (1/n²∑K_ss - 2/(nm)∑K_st) + (1/m²∑K_tt - 2/(nm)∑K_ts)
    # 注意：XY和YX已经是负值，所以用加法
    loss = (XX + XY).sum() + (YX + YY).sum()
    
    return loss


"""
=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `gdd`函数 | 公式(6) L_MMD | 最大均值差异损失 |
| `guassian_kernel` | 公式(6)中的φ映射 | 将特征映射到RKHS |
| `kernel_num=5` | 多核MMD | 使用5个不同带宽的核 |
| `XX, YY, XY, YX` | 公式(6)的展开 | 域内和域间相似度 |

=============================================================================
MMD数学原理详解
=============================================================================

1. **MMD的基本思想**：
   - 衡量两个分布P和Q是否相同
   - 如果P=Q，则所有统计矩都应该相同
   - MMD通过RKHS中的均值差异来度量

2. **公式推导**：
MMD² = || E_{x~P}[φ(x)] - E_{y~Q}[φ(y)] ||²_H
= E_{x,x'~P}[k(x,x')] + E_{y,y'~Q}[k(y,y')] - 2E_{x~P,y~Q}[k(x,y)]

text
其中k是核函数，φ是到RKHS的映射

3. **经验估计**（代码实现）：
MMD² ≈ 1/n²∑k(x_i,x_j) + 1/m²∑k(y_i,y_j) - 2/(nm)∑k(x_i,y_j)

text

=============================================================================
数据流示例
=============================================================================

假设：
- 源域特征 source: (10, 256)  # 10个样本，256维特征
- 目标域特征 target: (8, 256)  # 8个样本，256维特征

计算过程：
--------------------------------------------------------------------------------
1. guassian_kernel输出:
kernels: (18, 18)  # 10+8=18

2. 分割矩阵:
XX: (10, 10)  - 源域内部
YY: (8, 8)    - 目标域内部  
XY: (10, 8)   - 源域到目标域
YX: (8, 10)   - 目标域到源域

3. 计算各项:
XX = sum(XX)/(10*10) = 标量
YY = sum(YY)/(8*8)   = 标量
XY = -2*sum(XY)/(10*8) = 标量
YX = -2*sum(YX)/(8*10) = 标量

4. 最终loss:
loss = XX + YY + XY + YX  # 标量

=============================================================================
关键点解释
=============================================================================

1. **为什么用多核？**
- 单核可能无法捕捉所有尺度上的分布差异
- 多核MMD对不同频率的差异更鲁棒
- kernel_num=5表示用5个不同带宽的高斯核

2. **带宽（bandwidth）的作用？**
- 控制核函数的"敏感性"
- 小带宽：只关注很近的样本对
- 大带宽：考虑更远的样本关系
- 多尺度结合更全面

3. **为什么用高斯核？**
- 通用近似能力：可以逼近任意连续函数
- 数学性质好：正定核，保证MMD是有效的距离度量
- 计算方便：有闭式表达式

4. **MMD在HIA-Net中的作用？**
- 用于HRDA模块，对齐不同层的特征分布
- 减少源域和目标域之间的分布差异
- 帮助模型适应新被试的数据分布

=============================================================================
使用示例
=============================================================================

 """



# 注意：这个实现对应的是基础MMD
# 论文中的HRDA使用的是多层加权MMD：
# L_MMD = sum(w_i * L_MMD^i) 其中 w_i = alog(bi+1)
# text

# 这个MMD损失函数是HIA-Net中**领域自适应**的核心，它的作用是：

# 1. **分布对齐**：减少源域（已有被试）和目标域（新被试）的特征分布差异
# 2. **多核增强**：使用多个高斯核，更全面度量分布差异
# 3. **可微分**：可以作为损失函数的一部分进行端到端训练

