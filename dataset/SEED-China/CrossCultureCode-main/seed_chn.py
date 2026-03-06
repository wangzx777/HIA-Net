# -*- encoding: utf-8 -*-

import torch
import numpy as np
import pickle
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn import preprocessing

seed_chn_config = {
    'input_shape': (62, 5, 1),
    'num_class': 3
}

NUMBER_OF_SUBJECTS = 12
NUMBER_OF_SESSIONS = 3

NUMBER_OF_ALL_SESSIONS = 36

# file_path 
eeg_path = 'D:\Research\ML\Few-shot\SEED-China\\02-EEG-DE-feature\eeg_used_4s'
# eye_path = 'D:\Research\ML\Few-shot\SEED-China\\04-Eye-tracking-feature\eye_tracking_feature'

eeg_session_names = ['1_1.npz', '2_1.npz', '3_1.npz', '4_1.npz', '5_1.npz',
                      '8_1.npz', '9_1.npz', '10_1.npz', '11_1.npz', '12_1.npz',
                      '13_1.npz', '14_1.npz',
                     '1_2.npz', '2_2.npz', '3_2.npz', '4_2.npz', '5_2.npz',
                      '8_2.npz', '9_2.npz', '10_2.npz', '11_2.npz', '12_2.npz',
                      '13_2.npz', '14_2.npz',
                     '1_3.npz', '2_3.npz', '3_3.npz', '4_3.npz', '5_3.npz',
                      '8_3.npz', '9_3.npz', '10_3.npz', '11_3.npz', '12_3.npz',
                      '13_3.npz', '14_3.npz']

# eye_session_names = [['1_1', '2_1', '3_1', '4_1', '5_1',
#                       '8_1', '9_1', '10_1', '11_1', '12_1',
#                       '13_1', '14_1'],
#                      ['1_2', '2_2', '3_2', '4_2', '5_2',
#                       '8_2', '9_2', '10_2', '11_2', '12_2',
#                       '13_2', '14_2'],
#                      ['1_3', '2_3', '3_3', '4_3', '5_3',
#                       '8_3', '9_3', '10_3', '11_3', '12_3',
#                       '13_3', '14_3']]


def load_seed_chn(batch_size=None, shuffle=False, normalize='zscore'):

    tem_eeg_path = eeg_path
    for subject in tqdm(range(NUMBER_OF_SUBJECTS * NUMBER_OF_SESSIONS)):

        # load data of eeg_data
        npz_eeg = np.load(os.path.join(eeg_path, eeg_session_names[subject]))
        train_data_eeg = pickle.loads(npz_eeg['train_data'])
        test_data_eeg = pickle.loads(npz_eeg['test_data'])

        # train_label
        train_label = npz_eeg['train_label']
        # test_label
        test_label = npz_eeg['test_label']

        train_eeg_all_bands = []
        test_eeg_all_bands = []

        for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            train_tmp = train_data_eeg[bands]
            test_tmp = test_data_eeg[bands]
            if bands == 'delta':
                train_eeg_all_bands = train_tmp
                test_eeg_all_bands = test_tmp
            else:
                train_eeg_all_bands = np.hstack((train_eeg_all_bands, train_tmp))
                test_eeg_all_bands = np.hstack((test_eeg_all_bands, test_tmp))

    return train_eeg_all_bands, test_eeg_all_bands, train_label, test_label


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm


class SEEDChinaDataset(Dataset):
    def __init__(self, eeg_path, eeg_session_names, normalize='zscore'):
        self.eeg_path = eeg_path
        self.eeg_session_names = eeg_session_names
        self.normalize = normalize
        self.data = []
        self.labels = []
        self.NUMBER_OF_SUBJECTS = 15  # assuming 15 subjects
        self.NUMBER_OF_SESSIONS = 3  # assuming 3 sessions

        # Load all data into memory
        self._load_data()

    def _load_data(self):
        for subject in tqdm(range(self.NUMBER_OF_SUBJECTS * self.NUMBER_OF_SESSIONS)):
            npz_eeg = np.load(os.path.join(self.eeg_path, self.eeg_session_names[subject]))
            train_data_eeg = pickle.loads(npz_eeg['train_data'])
            test_data_eeg = pickle.loads(npz_eeg['test_data'])

            train_label = npz_eeg['train_label']
            test_label = npz_eeg['test_label']

            for data_eeg, labels in [(train_data_eeg, train_label), (test_data_eeg, test_label)]:
                all_bands_data = self._stack_bands(data_eeg)
                self.data.append(all_bands_data)
                self.labels.append(labels)

    def _stack_bands(self, data_eeg):
        """ Stack the EEG bands (delta, theta, alpha, beta, gamma) horizontally. """
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

        # Normalization if needed
        if self.normalize == 'zscore':
            sample = (sample - torch.mean(sample)) / torch.std(sample)

        return sample, label


# Usage example
def load_seed_chn(batch_size=32, shuffle=True, normalize='zscore'):
    eeg_path = "your/eeg/path"  # Update with actual path
    eeg_session_names = ["session1.npz", "session2.npz", ...]  # Fill with actual filenames

    dataset = SEEDChinaDataset(eeg_path, eeg_session_names, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


if __name__ == '__main__':
    train_data, test_data, train_label, test_label = load_seed_chn()

    print('train_data:', train_data.shape)
    print('test_data:', test_data.shape)

    print('train_label:',train_label.shape)
    print('test_label:', test_label.shape)

