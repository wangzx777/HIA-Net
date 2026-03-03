import torch.nn as nn
from torch.nn import functional as F
import torch


class ProtoNet(nn.Module):
    '''
    Prototypes of each class in support data
    原型网络 - 对应论文2.D节 Few-Shot Learning Objective
    
    功能：计算查询样本到各类原型的距离，用于Few-Shot分类
    论文公式：p_i = 1/|S_i| * sum(f(x))  # 原型计算
            p(y=i|x) = exp(-dist(f(x), p_i)) / sum(exp(-dist(f(x), p_j)))  # 分类概率
    '''

    def __init__(self):
        super(ProtoNet, self).__init__()

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        计算欧氏距离 - 对应论文公式(7)中的dist函数
        
        Args:
            x: 查询样本特征，形状 (n_classes*n_query, dim)
            y: 原型特征，形状 (n_classes, dim)
        
        Returns:
            距离矩阵，形状 (n_classes*n_query, n_classes)
        '''
        # x(queries): N x D (n_classes*n_query, dim)
        # y(prototypes): M x D (n_classes, dim)
        n = x.size(0)  # 查询样本总数
        m = y.size(0)  # 类别数
        d = x.size(1)  # 特征维度
        if d != y.size(1):
            raise Exception

        # 扩展维度以便计算所有配对距离
        x = x.unsqueeze(1).expand(n, m, d)  # (n_classes*n_query, n_classes, dim)
        y = y.unsqueeze(0).expand(n, m, d)  # (n_classes*n_query, n_classes, dim)

        # 计算欧氏距离的平方
        return torch.pow(x - y, 2).sum(2)  # (n_classes*n_query, n_classes)

    def forward(self, x, y, n_classes, n_support=None, n_query=None, flag=0):
        """
        前向传播 - 计算查询样本到原型的距离
        
        Args:
            x: 特征数据
            y: 标签
            n_classes: 类别数N（论文中N=3：Positive, Neutral, Negative）
            n_support: support集样本数K（论文中K=1,5,10,20）
            n_query: query集样本数（论文中每个类20个）
            flag: 0表示support和query都来自同一域（源域训练）
                  1表示support来自源域，query来自目标域（测试）
        
        Returns:
            dists: 距离矩阵，形状 (n_classes*n_query, n_classes)
        """
        if flag == 0:  # support and query all come from source/target
            # 训练模式：support和query都来自源域
            def supp_idxs(c):
                """
                为每个类 c 提取前 n_support 个样本的索引

                假设 y = torch.tensor([0, 1, 2, 1, 2, 0, 1]) 且 n_support = 2，函数的作用如下：
                如果调用 supp_idxs(1)，那么：
                y.eq(1) 返回 [False, True, False, True, False, False, True]
                .nonzero() 返回 [1, 3, 6]
                [:n_support] 截取前 2 个，结果为 [1, 3]
                .squeeze(1) 后，得到 [1, 3]，表示标签为 1 的前两个样本的索引。
                """
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            classes = torch.unique(y)  # 获取所有类别 [0,1,2]
            # 为每个类 c 提取前 n_support 个样本的索引
            support_idxs = list(map(supp_idxs, classes))
            # 为每个类 c 提取剩下的样本作为query
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
            
            # 根据索引提取特征
            query_samples = x[query_idxs]  # (n_classes * n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

        else:  # support and query come from source and target respectively
            # 测试模式：support来自源域，query来自目标域
            # 对应论文：在目标域中随机选择K个标记样本作为support集
            data_src, data_tgt = x[0], x[1]  # 源域和目标域数据
            label_src, label_tgt = y[0], y[1]  # 源域和目标域标签
            classes = torch.unique(label_src)  # 获取所有类别
            
            # 从源域中提取support样本（每个类前n_support个）
            support_idxs = list(map(lambda c: label_src.eq(c).nonzero()[:n_support], classes))
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, n_support, -1)  # (n_classes, n_support, dim)
            
            # 从目标域中提取query样本（每个类前n_query个）
            query_idxs = list(map(lambda c: label_tgt.eq(c).nonzero()[:n_query], classes))
            query_idxs = torch.stack(query_idxs).view(-1)
            query_samples = data_tgt[query_idxs]  # (n_classes*n_query, dim)

        # 对每个类别取平均值作为原型 - 对应论文公式：p_i = 1/|S_i| * sum(f(x))
        support_prototypes = support_samples.mean(1)  # (n_classes, dim)
        
        # 返回每个query到原型的欧氏距离 - 用于后续计算分类概率
        # 对应论文公式(7)中的 -dist(f(x), p_i)
        dists = self.euclidean_dist(query_samples, support_prototypes)
        return dists


"""
=============================================================================
数据流形状变化追踪
=============================================================================

训练模式 (flag=0):
------------------------------------------------------------------------------
输入:
    x: (total_samples, dim)  - 所有样本的特征
    y: (total_samples,)       - 对应的标签
    
处理过程:
1. 提取support索引和query索引
   support_idxs: (n_classes * n_support,)
   query_idxs:   (n_classes * n_query,)

2. 根据索引提取特征
   support_samples_raw: (n_classes * n_support, dim)
   query_samples:       (n_classes * n_query, dim)

3. reshape support样本
   support_samples: (n_classes, n_support, dim)
        ↓ mean(dim=1)  # 对每个类的support样本取平均
   support_prototypes: (n_classes, dim)

4. 计算欧氏距离
   query_samples: (n_classes*n_query, dim)
   support_prototypes: (n_classes, dim)
        ↓ euclidean_dist
   dists: (n_classes*n_query, n_classes)

测试模式 (flag=1):
------------------------------------------------------------------------------
输入:
    x[0]: (src_samples, dim)  - 源域特征
    x[1]: (tgt_samples, dim)  - 目标域特征
    y[0]: (src_samples,)       - 源域标签
    y[1]: (tgt_samples,)       - 目标域标签

处理过程:
1. 提取support索引（从源域）
   support_idxs: (n_classes * n_support,)
   support_samples_raw: (n_classes * n_support, dim)
        ↓ view(n_classes, n_support, dim)
   support_samples: (n_classes, n_support, dim)
        ↓ mean(dim=1)
   support_prototypes: (n_classes, dim)

2. 提取query索引（从目标域）
   query_idxs: (n_classes * n_query,)
   query_samples: (n_classes * n_query, dim)

3. 计算欧氏距离
   dists: (n_classes*n_query, n_classes)


典型数值示例 (3-way 5-shot):
------------------------------------------------------------------------------
n_classes = 3 (Positive, Neutral, Negative)
n_support = 5
n_query = 20

训练模式:
- x形状: (75, dim)  [3类 × (5支持+20查询) = 75个样本]
- support_samples: (3, 5, dim)
- query_samples: (60, dim)  [3类 × 20查询 = 60]
- support_prototypes: (3, dim)
- dists: (60, 3)  [60个查询样本到3个原型的距离]

测试模式:
- data_src: (src_samples, dim)  [源域所有样本]
- data_tgt: (tgt_samples, dim)  [目标域所有样本]
- support_samples: (3, 5, dim)
- query_samples: (60, dim)
- dists: (60, 3)

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `ProtoNet` | 2.D节 Few-Shot Learning Objective | 基于原型的Few-Shot学习 |
| `support_samples.mean(1)` | 公式 p_i = 1/|S_i| * sum(f(x)) | 计算每个类的原型 |
| `euclidean_dist` | 公式(7)中的dist函数 | 计算欧氏距离 |
| `flag=0`模式 | 训练阶段 | 从源域构造episode |
| `flag=1`模式 | 测试/验证阶段 | support源域，query目标域 |
| `n_classes=3` | 表I | 3种情感：Positive, Neutral, Negative |
| `n_support=K` | 实验设置 | K=1,5,10,20 shot设置 |
| `n_query=20` | 实验设置 | 每个类20个查询样本 |

=============================================================================
关键点解释
=============================================================================

1. **为什么需要两种flag模式？**
   - flag=0: 训练时，support和query都来自源域，模拟episode学习
   - flag=1: 测试时，support来自源域（有标签），query来自目标域（预测）
   - 对应论文：在目标域中随机选择K个标记样本作为support集

2. **原型计算为什么用mean？**
   - 原型网络的核心思想：每个类的原型是该类所有support样本的均值
   - 在特征空间中，同类样本应该聚集在原型周围

3. **为什么返回距离而不是概率？**
   - 返回距离，然后在外部通过softmax计算概率
   - 对应论文公式(7)：p = exp(-dist) / sum(exp(-dist))
   - 这样设计更灵活，损失函数可以在外部计算

4. **索引选择逻辑的作用？**
   - 确保每个类在support和query中都有代表性的样本
   - 符合Few-Shot学习的episode采样方式
"""