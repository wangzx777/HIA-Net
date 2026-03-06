# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    """
    原型批量采样器 - 对应论文2.D节的episode采样策略
    
    功能：在每个iteration（episode）中，从数据集中随机选择N个类别（N-way），
         每个类别随机选择K个样本（K-shot），组成一个batch用于Few-Shot训练
    
    Few-Shot episode结构：
    - 每个episode包含N个类别，每个类别K个样本
    - 总batch_size = N * K
    - 这些样本会进一步划分为support set和query set
    
    论文3.B节："In an N-way K-shot setting, the support set contains N×K labeled samples"
    """
    def __init__(self, labels, classes_per_it, num_samples, iterations):
        """
        Args:
            labels: 数据集的所有标签，形状 (total_samples,)
            classes_per_it: 每个episode中的类别数N（论文中N=3）
            num_samples: 每个类别采样的样本数K（论文中K=1,5,10,20）
            iterations: 每个epoch中的episode数量（论文中=20）
        
        注意：这里的num_samples是每个类的总采样数，后续会划分为support和query
             例如在prototypical_loss中，前n_support个作为support，其余作为query
        """
        self.labels = labels
        self.classes_per_it = classes_per_it  # N-way
        self.sample_per_class = num_samples    # K-shot（实际是support+query的总数）
        self.iterations = iterations            # 每个epoch的episode数
        
        # 获取所有唯一的类别及其出现次数
        self.classes, self.counts = np.unique(labels, return_counts=True)

    def __iter__(self):
        """使采样器成为可迭代对象"""
        return self

    def __next__(self):
        """
        生成一个episode的样本索引
        
        Returns:
            batch_indices: 选中的样本索引，形状 (N * K,)
        
        采样策略：
        1. 随机选择N个类别
        2. 对每个选中的类别，随机选择K个样本
        3. 将所有选中的索引拼接返回
        
        数据流示例 (3-way 5-shot):
        - 类别总数: 假设有10个类别
        - 随机选择3个类别: [2, 5, 8]
        - 对类别2: 随机选择5个样本索引 [12, 34, 56, 78, 90]
        - 对类别5: 随机选择5个样本索引 [23, 45, 67, 89, 101]
        - 对类别8: 随机选择5个样本索引 [32, 54, 76, 98, 110]
        - 拼接: [12,34,56,78,90,23,45,67,89,101,32,54,76,98,110]
        """
        spc = self.sample_per_class  # 每个类的采样数K
        cpi = self.classes_per_it     # 类别数N
        batch_size = spc * cpi        # 总batch大小
        batch_indices = []

        # 步骤1：随机选择N个类别
        # torch.randperm(len(self.classes)) 生成随机排列的索引
        # [:cpi] 取前N个，实现无放回随机选择
        c_idxs = torch.randperm(len(self.classes))[:cpi]

        # 步骤2：对每个选中的类别，随机选择K个样本
        for i, c in enumerate(self.classes[c_idxs]):
            # 创建该类别的掩码，找出所有属于该类别的样本
            label_mask = self.labels == c
            sample_indices = torch.nonzero(label_mask).squeeze()  # 获取该类别的所有索引
            
            # 处理特殊情况：如果该类别只有一个样本，确保维度正确
            if sample_indices.dim() == 0:  # 如果是标量张量
                sample_indices = sample_indices.unsqueeze(0)  # 转为1维张量
            
            # 从该类别中随机选择K个样本
            # torch.randperm(len(sample_indices)) 随机排列索引
            # [:spc] 取前K个
            selected_samples = sample_indices[torch.randperm(len(sample_indices))[:spc]]

            # 将选中的样本索引加入批次
            batch_indices.append(selected_samples)

        # 步骤3：合并所有选中的索引
        batch_indices = torch.cat(batch_indices)

        # 返回EEG和ET数据共用的索引（因为EEG和ET是配对的对齐数据）
        return batch_indices

    def __len__(self):
        """返回每个epoch的episode数量"""
        return self.iterations


"""
=============================================================================
被注释掉的旧版本代码分析
=============================================================================

下面是原始实现（被注释掉的部分），功能相同但实现更复杂。
它预先构建了一个索引矩阵，提前计算好每个类别的所有样本索引。
当前版本更简洁高效，直接动态采样。
"""

# class PrototypicalBatchSampler(object):
#     '''
#     PrototypicalBatchSampler: yield a batch of indexes at each iteration.
#     Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
#     In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
#     for 'classes_per_it' random classes.
#
#     __len__ returns the number of episodes per epoch (same as 'self.iterations').
#     '''
#
#     def __init__(self, labels, classes_per_it, num_samples, iterations):
#         '''
#         Initialize the PrototypicalBatchSampler object
#         Args:
#         - labels: an iterable containing all the labels for the current dataset
#         samples indexes will be infered from this iterable.
#         - classes_per_it: number of random classes for each iteration
#         - num_samples: number of samples for each iteration for each class (support + query)
#         - iterations: number of iterations (episodes) per epoch
#         '''
#         super(PrototypicalBatchSampler, self).__init__()
#         self.labels = labels
#         self.classes_per_it = classes_per_it
#         self.sample_per_class = num_samples
#         self.iterations = iterations
#         # print("sample_per_class:", self.sample_per_class)
#         self.classes, self.counts = np.unique(self.labels, return_counts=True)
#         self.classes = torch.LongTensor(self.classes)
#
#         # 创建索引矩阵：行数=类别数，列数=每个类别的最大样本数
#         # 用nan填充空白位置
#         self.idxs = range(len(self.labels))
#         self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
#         self.indexes = torch.Tensor(self.indexes)
#         self.numel_per_class = torch.zeros_like(self.classes)
#         
#         # 填充索引矩阵：为每个类别记录其所有样本的索引
#         for idx, label in enumerate(self.labels):
#             label_idx = np.argwhere(self.classes == label).item()
#             # 找到该类别第一个空位置填入索引
#             self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
#             self.numel_per_class[label_idx] += 1
#         # print('number per class: ', self.numel_per_class)
#
#     def __iter__(self):
#         '''
#         yield a batch of indexes
#         '''
#         spc = self.sample_per_class
#         cpi = self.classes_per_it
#
#         for it in range(self.iterations):
#             batch_size = spc * cpi
#             batch = torch.LongTensor(batch_size)
#             c_idxs = torch.randperm(len(self.classes))[:cpi]
#             for i, c in enumerate(self.classes[c_idxs]):
#                 s = slice(i * spc, (i + 1) * spc)
#                 # FIXME when torch.argwhere will exists
#                 label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
#                 sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
#                 # print("label_idx:", label_idx)
#                 # print("len(self.indexes[label_idx]):", len(self.indexes[label_idx]))
#                 # print("self.numel_per_class[label_idx]:", self.numel_per_class[label_idx])
#                 # print("spc:", spc)
#                 # print(len(sample_idxs))
#                 batch[s] = self.indexes[label_idx][sample_idxs]
#             batch = batch[torch.randperm(len(batch))]
#             yield batch
#
#     def __len__(self):
#         '''
#         returns the number of iterations (episodes) per epoch
#         '''
#         return self.iterations


"""
=============================================================================
数据流示例
=============================================================================

假设：
- 数据集：1000个样本，10个类别（0-9）
- labels: [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...] 每个类别100个样本
- classes_per_it (N) = 3
- sample_per_class (K) = 5
- iterations = 20

__next__() 调用一次输出：
--------------------------------------------------------------------------------
c_idxs = torch.randperm(10)[:3] = [2, 7, 4]  # 随机选中类别2,7,4

对类别2：
    label_mask = [False,False,True,...]  # 标记类别2的样本
    sample_indices = [200,201,202,...,299]  # 类别2的所有100个索引
    selected_samples = [245, 201, 278, 234, 267]  # 随机选5个

对类别7：
    selected_samples = [712, 734, 756, 789, 701]  # 类别7的5个索引

对类别4：
    selected_samples = [412, 434, 456, 489, 401]  # 类别4的5个索引

batch_indices = [245,201,278,234,267, 712,734,756,789,701, 412,434,456,489,401]
形状: (15,)

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `PrototypicalBatchSampler` | 2.D节 episode采样 | 每个iteration生成N-way K-shot的batch |
| `classes_per_it` | 3.B节 N-way | 每个episode的类别数，论文N=3 |
| `sample_per_class` | 3.B节 K-shot | 每个类的样本数，论文K=1,5,10,20 |
| `iterations` | 3.B节 "20 episodes" | 每个epoch的episode数 |
| 随机类别选择 | Few-Shot标准做法 | 保证每个episode的类别多样性 |
| 随机样本选择 | Few-Shot标准做法 | 保证每个episode的样本多样性 |

=============================================================================
关键点解释
=============================================================================

1. **为什么需要这个采样器？**
   - Few-Shot Learning不是用传统的数据加载方式（随机batch）
   - 需要专门构造episode：每个episode包含N个类别，每类K个样本
   - 模拟测试时的场景（只有少量标记样本）

2. **sample_per_class和n_support的关系？**
   - 这里的sample_per_class是每个类采样的总样本数
   - 在prototypical_loss中，前n_support个作为support，其余作为query
   - 所以 sample_per_class = n_support + n_query

3. **为什么返回索引而不是直接返回数据？**
   - 采样器只负责提供索引，不负责加载实际数据
   - 解耦了采样和数据加载
   - 便于后续灵活使用（EEG和眼动共用索引）

4. **如何保证每个episode的类别平衡？**
   - 每个类别采样相同数量的样本（sample_per_class）
   - 类别随机选择，但选中后每个类采样数相同
   - 避免类别不平衡问题

5. **新版本 vs 旧版本实现？**
   - 旧版本：预计算索引矩阵，提前分配内存
   - 新版本：动态采样，更简洁，内存效率更高
   - 新版本更容易理解和维护

=============================================================================
在训练循环中的使用
=============================================================================

```python
# 初始化采样器
train_sampler = PrototypicalBatchSampler(
    labels=train_labels,           # 训练集所有标签
    classes_per_it=3,              # 3-way
    num_samples=5+20,              # 5 support + 20 query = 25个样本/类
    iterations=20                   # 每个epoch 20个episode
)

# 创建DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,   # 使用自定义采样器
    num_workers=4
)

# 训练循环
for epoch in range(50):
    for batch_idx, (eeg, eye, labels) in enumerate(train_loader):
        # batch的形状:
        # eeg: (batch_size, 5, 9, 9) 其中batch_size = 3 * 25 = 75
        # eye: (batch_size, 33)
        # labels: (batch_size,)
        
        # 在prototypical_loss中，前5个样本/类作为support
        # 后20个样本/类作为query
        loss = prototypical_loss(model_output, labels, n_support=5)
"""