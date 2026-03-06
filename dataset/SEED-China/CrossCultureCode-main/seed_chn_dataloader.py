import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm


class SEEDChinaDataset(Dataset):
    def __init__(self, eeg_path, eeg_session_names, train=True, normalize='zscore'):
        self.eeg_path = eeg_path
        self.eeg_session_names = eeg_session_names
        self.normalize = normalize
        self.train = train
        self.data = []
        self.labels = []
        self.NUMBER_OF_SUBJECTS = 12
        self.NUMBER_OF_SESSIONS = 3

        self._load_data()

    def _load_data(self):
        for subject in tqdm(range(self.NUMBER_OF_SUBJECTS * self.NUMBER_OF_SESSIONS)):
            npz_eeg = np.load(os.path.join(self.eeg_path, self.eeg_session_names[subject]))

            # 根据self.train加载训练数据或测试数据
            if self.train:
                data_eeg = pickle.loads(npz_eeg['train_data'])
                labels = npz_eeg['train_label']
            else:
                data_eeg = pickle.loads(npz_eeg['test_data'])
                labels = npz_eeg['test_label']

            all_bands_data = self._stack_bands(data_eeg)

            # 将所有样本和标签分别加入列表
            self.data.extend(all_bands_data)  # 将 499 个样本添加到 self.data
            self.labels.extend(labels)  # 将 499 个标签添加到 self.labels

    def _stack_bands(self, data_eeg):
        all_bands = None
        for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            if all_bands is None:
                all_bands = data_eeg[bands]
            else:
                all_bands = np.hstack((all_bands, data_eeg[bands]))
        return all_bands

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.normalize == 'zscore':
            sample = (sample - torch.mean(sample)) / torch.std(sample)
        return sample, label


# # Usage example
# def load_seed_chn(batch_size=32, shuffle=True, normalize='zscore'):
#     eeg_path = "D:\Research\ML\Few-shot\SEED-China\\02-EEG-DE-feature\eeg_used_4s"  # Update with actual path
#
#     eeg_session_names = ['1_1.npz', '2_1.npz', '3_1.npz', '4_1.npz', '5_1.npz',
#                          '8_1.npz', '9_1.npz', '10_1.npz', '11_1.npz', '12_1.npz',
#                          '13_1.npz', '14_1.npz',
#                          '1_2.npz', '2_2.npz', '3_2.npz', '4_2.npz', '5_2.npz',
#                          '8_2.npz', '9_2.npz', '10_2.npz', '11_2.npz', '12_2.npz',
#                          '13_2.npz', '14_2.npz',
#                          '1_3.npz', '2_3.npz', '3_3.npz', '4_3.npz', '5_3.npz',
#                          '8_3.npz', '9_3.npz', '10_3.npz', '11_3.npz', '12_3.npz',
#                          '13_3.npz', '14_3.npz']
#
#     dataset = SEEDChinaDataset(eeg_path, eeg_session_names, normalize=normalize)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
#     return dataloader
