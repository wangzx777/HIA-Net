# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    def __init__(self, labels, classes_per_it, num_samples, iterations):
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        self.classes, self.counts = np.unique(labels, return_counts=True)

    def __iter__(self):
        return self

    def __next__(self):
        spc = self.sample_per_class
        cpi = self.classes_per_it
        batch_size = spc * cpi
        batch_indices = []


        c_idxs = torch.randperm(len(self.classes))[:cpi]

        for i, c in enumerate(self.classes[c_idxs]):
            label_mask = self.labels == c
            sample_indices = torch.nonzero(label_mask).squeeze()  # 获取该类别的所有索引
            if sample_indices.dim() == 0:  # 如果是标量张量
                sample_indices = sample_indices.unsqueeze(0)  # 转为1维张量
            selected_samples = sample_indices[torch.randperm(len(sample_indices))[:spc]]  # 随机选择样本

            # 将选中的样本索引加入批次
            batch_indices.append(selected_samples)  # 假设 EEG 数据的索引

        # 合并索引
        batch_indices = torch.cat(batch_indices)

        # 返回 EEG 和 ET 数据的索引
        return batch_indices

    def __len__(self):
        return self.iterations


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
#         # create a matrix, indexes, of dim: classes X max(elements per class)
#         # fill it with nans
#         # for every class c, fill the relative row with the sample's indices in labels belonging to c
#         # in numel_per_class we store the number of samples for each class/row
#         self.idxs = range(len(self.labels))
#         self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
#         self.indexes = torch.Tensor(self.indexes)
#         self.numel_per_class = torch.zeros_like(self.classes)
#         for idx, label in enumerate(self.labels):
#             label_idx = np.argwhere(self.classes == label).item()
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


