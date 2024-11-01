import os, sys
import pickle

import pandas as pd
import torch
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def manual_split(eeg_data, eye_data, slabels, num_support_per_class, classes_per_it):
    # 获取eeg_data的原始形状
    original_eeg_shape = eeg_data.shape  # 假设为 (834, 5, 9, 9)
    eeg_sample_shape = original_eeg_shape[1:]  # 保存除去batch size外的维度 (5, 9, 9)

    # 将eeg_data展平成2维数组
    df_eeg = pd.DataFrame(eeg_data.reshape(eeg_data.shape[0], -1))
    df_eye = pd.DataFrame(eye_data)
    df_labels = pd.Series(slabels)

    train_data_eeg = []
    temp_data_eeg = []
    train_data_eye = []
    temp_data_eye = []
    train_labels = []
    temp_labels = []

    # 遍历每个类别
    for label in np.unique(slabels):
        # 选出属于该类别的数据
        class_eeg = df_eeg[df_labels == label]
        class_eye = df_eye[df_labels == label]
        class_labels = df_labels[df_labels == label]

        # 随机打乱数据
        shuffled_idx = np.random.permutation(len(class_eeg))
        class_eeg = class_eeg.iloc[shuffled_idx]
        class_eye = class_eye.iloc[shuffled_idx]
        class_labels = class_labels.iloc[shuffled_idx]

        # 分割出指定数量的支持样本
        train_data_eeg.append(class_eeg[:num_support_per_class].values)
        temp_data_eeg.append(class_eeg[num_support_per_class:].values)
        train_data_eye.append(class_eye[:num_support_per_class].values)
        temp_data_eye.append(class_eye[num_support_per_class:].values)
        train_labels.append(class_labels[:num_support_per_class].values)
        temp_labels.append(class_labels[num_support_per_class:].values)

    # 将分割的数据合并
    train_data_eeg = np.vstack(train_data_eeg)
    temp_data_eeg = np.vstack(temp_data_eeg)
    train_data_eye = np.vstack(train_data_eye)
    temp_data_eye = np.vstack(temp_data_eye)
    train_labels = np.concatenate(train_labels)
    temp_labels = np.concatenate(temp_labels)

    # 恢复eeg_data的原始形状
    train_data_eeg = train_data_eeg.reshape(-1, *eeg_sample_shape)
    temp_data_eeg = temp_data_eeg.reshape(-1, *eeg_sample_shape)

    return train_data_eeg, temp_data_eeg, train_data_eye, temp_data_eye, train_labels, temp_labels



def load_data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,sess_idx, idx, mode):
    # load the label data
    eeg_sample = []
    eye_sample = []
    labels = []

    for i in tqdm(idx):
        # print(f"sess_idx:{sess_idx}")
        # print(f"i:{i}")
        npz_eeg = np.load(os.path.join(eeg_path, eeg_session_names[sess_idx][i]))
        eye_data = pickle.load(open(os.path.join(eye_path, eye_session_names[sess_idx][i]), 'rb'))

        # 根据self.train加载训练数据或测试数据
        train_data_eye = np.asarray(eye_data['train_data_eye'].tolist()).squeeze()

        #normalize eye
        scaler = preprocessing.MinMaxScaler()
        train_data_eye = scaler.fit_transform(train_data_eye)


        train_data_eeg = pickle.loads(npz_eeg['train_data'])
        train_labels = npz_eeg['train_label']

        # normalize tensor
        train_data_eeg = standardize_bands(train_data_eeg)

        # reshape the tensor
        train_data_eeg = exact_bands(train_data_eeg)

        # 1d channel vector to 2d channel matrix
        train_data_eeg = convert_chl(train_data_eeg)  # (499, 5, 9, 9)


        test_data_eye = np.asarray(eye_data['test_data_eye'].tolist()).squeeze()
        test_data_eye = scaler.fit_transform(test_data_eye)
        test_data_eeg = pickle.loads(npz_eeg['test_data'])
        test_labels = npz_eeg['test_label']
        test_data_eeg = standardize_bands(test_data_eeg)
        test_data_eeg = exact_bands(test_data_eeg)
        test_data_eeg = convert_chl(test_data_eeg)  # (499, 5, 9, 9)



        if mode == 'full':
            eeg_sample.extend(train_data_eeg)
            eye_sample.extend(train_data_eye)
            labels.extend(train_labels)

            eeg_sample.extend(test_data_eeg)
            eye_sample.extend(test_data_eye)
            labels.extend(test_labels)


        else:
            eeg_data = np.concatenate((train_data_eeg, test_data_eeg), axis=0)
            eye_data = np.concatenate((train_data_eye, test_data_eye), axis=0)
            slabels = np.concatenate((train_labels, test_labels), axis=0)

            # 检查数据是否有相同的样本数
            assert eeg_data.shape[0] == eye_data.shape[0] == slabels.shape[0], "样本数不一致"

            # 生成一个打乱的索引顺序
            indices = np.random.permutation(eeg_data.shape[0])

            # 按照打乱的索引顺序重排数据
            eeg_data = eeg_data[indices]
            eye_data = eye_data[indices]
            slabels = slabels[indices]

            train_data_eeg, temp_data_eeg, train_data_eye, temp_data_eye, train_labels, temp_labels = manual_split(
                eeg_data, eye_data, slabels,
                num_support_per_class=parser.num_support_src,
                classes_per_it=parser.classes_per_it_src
            )

            val_data_eeg, test_data_eeg, val_data_eye, test_data_eye, val_labels, test_labels = train_test_split(
                temp_data_eeg, temp_data_eye, temp_labels,
                train_size=300, random_state=42,
                stratify=temp_labels)

            if mode == 'train':
                # 将train_data_eeg, train_data_eye, 和train_labels进行拆分

                eeg_sample.extend(train_data_eeg)
                eye_sample.extend(train_data_eye)
                labels.extend(train_labels)

            if mode == 'val':
                eeg_sample.extend(val_data_eeg)
                eye_sample.extend(val_data_eye)
                labels.extend(val_labels)

            if mode == 'test':
                eeg_sample.extend(test_data_eeg)
                eye_sample.extend(test_data_eye)
                labels.extend(test_labels)

    return eeg_sample, eye_sample, labels



# def normalize(features, select_dim=0):
#     features_min, _ = torch.min(features, dim=select_dim)
#     features_max, _ = torch.max(features, dim=select_dim)
#     features_min = features_min.unsqueeze(select_dim)
#     features_max = features_max.unsqueeze(select_dim)
#     return (features - features_min) / (features_max - features_min)


def reshape_feature(fts_sample):
    fts = None
    if torch.is_tensor(fts_sample):
        elec, number, freq = fts_sample.shape
        fts = fts_sample.permute(1, 0, 2).reshape(number, elec * freq)
    else:
        print("Something Wrong!!!")

    return fts

def stack_bands(data_eeg):
    all_bands = []
    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        all_bands.append(data_eeg[bands])  # 每个频段形状为 (499, 62)

    # 使用 np.stack 在新轴上拼接频段，axis=1 表示在第1个维度上拼接
    all_bands = np.concatenate(all_bands, axis=1)  # 最终形状为 (499, 62 * 5)
    return all_bands

def exact_bands(data_eeg):
    all_bands = []
    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        all_bands.append(data_eeg[bands])  # 每个频段形状为 (499, 62)

    # 使用 np.stack 在新轴上拼接频段，axis=2 表示在第2个维度上拼接
    all_bands = np.stack(all_bands, axis=2)  # 最终形状为 (499, 62 , 5)
    return all_bands

def standardize_bands(data_eeg):
    standardized_bands = {}

    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        scaler = StandardScaler()
        # Apply StandardScaler to each band's data across the 2D array dimensions
        standardized_bands[band] = scaler.fit_transform(data_eeg[band])

    return standardized_bands
def data_1Dto2D_62chl(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          data[0],    data[1],    data[2], 	0,  	    0, 	        0      )
    data_2D[1] = (0,  	   	0,          0,          data[3],    0,          data[4],    0,          0,          0       )
    data_2D[2] = (data[5],  data[6],    data[7],    data[8],    data[9],    data[10],   data[11],   data[12],   data[13])
    data_2D[3] = (data[14], data[15],   data[16],   data[17],   data[18],   data[19],   data[20],   data[21],   data[22])
    data_2D[4] = (data[23], data[24],   data[25],   data[26],   data[27],   data[28],   data[29],   data[30],   data[31])
    data_2D[5] = (data[32], data[33],   data[34],   data[35],   data[36],   data[37],   data[38],   data[39],   data[40])
    data_2D[6] = (data[41], data[42],   data[43],   data[44],   data[45],   data[46],   data[47],   data[48],   data[49])
    data_2D[7] = (0,        data[50],   data[51],   data[52],   data[53],   data[54],   data[55],   data[56],   0       )
    data_2D[8] = (0,        0,          data[57],   data[58],   data[59],   data[60],   data[61],   0,       0       )
    # return shape:9*9
    return data_2D

def convert_chl(data):
    num_samples, num_channels, num_bands = data.shape
    data_2d_all = np.zeros((num_samples, num_bands, 9, 9))
    for i in range(num_samples):
        for j in range(num_bands):
            data_1d = data[i, :, j]
            data_2d = data_1Dto2D_62chl(data_1d)
            data_2d_all[i, j, :, :] = data_2d
    return data_2d_all  # (num_samples, num_bands, 9, 9)

def load4data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,sess_idx, idx,mode):
    """
    load the SEED data set for TL
    """
    eeg_sample, eye_sample, labels = load_data(parser,eeg_path, eye_path,eeg_session_names,eye_session_names,sess_idx, idx,mode)
    # transfer from ndarray to tensor

    eeg_sample = np.array(eeg_sample)
    eeg_sample = torch.from_numpy(eeg_sample).type(torch.FloatTensor)

    eye_sample = np.array(eye_sample)
    eye_sample = torch.from_numpy(eye_sample).type(torch.FloatTensor)


    labels = np.array(labels)
    labels = torch.from_numpy(labels).type(torch.LongTensor)

    # 1d channel vector to 2d channel matrix
    # data_sample = convert_chl(data_sample)  # (12 * 499, 5, 9, 9)
    # data_sample = data_sample.astype('float32')

    return eeg_sample, eye_sample, labels
