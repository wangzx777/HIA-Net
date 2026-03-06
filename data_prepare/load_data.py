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
    """
    手动划分support集和剩余集（用于构造Few-Shot episode）
    
    功能：从每个类别中随机抽取指定数量的样本作为support集，其余作为临时集
    对应论文：2.D节中的support set构建
    
    Args:
        eeg_data: EEG数据，形状 (total_samples, 5, 9, 9)
        eye_data: 眼动数据，形状 (total_samples, 33)
        slabels: 标签，形状 (total_samples,)
        num_support_per_class: 每个类的support样本数K
        classes_per_it: 类别数N（论文中为3）
    
    Returns:
        train_data_eeg: support集的EEG数据
        temp_data_eeg: 剩余集的EEG数据
        train_data_eye: support集的眼动数据
        temp_data_eye: 剩余集的眼动数据
        train_labels: support集的标签
        temp_labels: 剩余集的标签
    """
    # 获取eeg_data的原始形状
    original_eeg_shape = eeg_data.shape  # 假设为 (834, 5, 9, 9)
    eeg_sample_shape = original_eeg_shape[1:]  # 保存除去batch size外的维度 (5, 9, 9)

    # 将eeg_data展平成2维数组，便于用pandas操作
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

        # 分割出指定数量的支持样本（前num_support_per_class个）
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

    # 恢复eeg_data的原始形状（从展平状态恢复）
    train_data_eeg = train_data_eeg.reshape(-1, *eeg_sample_shape)
    temp_data_eeg = temp_data_eeg.reshape(-1, *eeg_sample_shape)

    return train_data_eeg, temp_data_eeg, train_data_eye, temp_data_eye, train_labels, temp_labels


def load_data(parser, eeg_path, eye_path, eeg_session_names, eye_session_names, sess_idx, idx, mode):
    """
    加载单个被试的EEG和眼动数据
    
    Args:
        parser: 配置参数
        eeg_path: EEG数据路径
        eye_path: 眼动数据路径
        eeg_session_names: EEG会话文件名列表
        eye_session_names: 眼动会话文件名列表
        sess_idx: 会话索引（SEED有3个session）
        idx: 被试索引列表
        mode: 'full'、'train'、'val'或'test'
    
    Returns:
        eeg_sample: EEG样本列表
        eye_sample: 眼动样本列表
        labels: 标签列表
    """
    eeg_sample = []
    eye_sample = []
    labels = []

    for i in tqdm(idx):
        # 加载EEG数据（.npz格式）
        npz_eeg = np.load(os.path.join(eeg_path, eeg_session_names[sess_idx][i]))
        
        # 加载眼动数据（.pkl格式）
        eye_data = pickle.load(open(os.path.join(eye_path, eye_session_names[sess_idx][i]), 'rb'))

        # ============ 处理训练集数据 ============
        # 提取眼动训练数据并归一化
        train_data_eye = np.asarray(eye_data['train_data_eye'].tolist()).squeeze()
        scaler = preprocessing.MinMaxScaler()  # MinMax归一化到[0,1]
        train_data_eye = scaler.fit_transform(train_data_eye)

        # 提取EEG训练数据
        train_data_eeg = pickle.loads(npz_eeg['train_data'])
        train_labels = npz_eeg['train_label']

        # EEG预处理：标准化 -> 提取频带 -> 转2D拓扑图
        train_data_eeg = standardize_bands(train_data_eeg)  # 每个频带标准化
        train_data_eeg = exact_bands(train_data_eeg)        # 提取5个频带
        train_data_eeg = convert_chl(train_data_eeg)        # 转9×9 2D矩阵 (499, 5, 9, 9)

        # ============ 处理测试集数据 ============
        test_data_eye = np.asarray(eye_data['test_data_eye'].tolist()).squeeze()
        test_data_eye = scaler.fit_transform(test_data_eye)  # 注意：这里重新fit了，可能有问题
        
        test_data_eeg = pickle.loads(npz_eeg['test_data'])
        test_labels = npz_eeg['test_label']
        
        # 同样的EEG预处理
        test_data_eeg = standardize_bands(test_data_eeg)
        test_data_eeg = exact_bands(test_data_eeg)
        test_data_eeg = convert_chl(test_data_eeg)  # (499, 5, 9, 9)

        # ============ 根据mode组织数据 ============
        if mode == 'full':
            # 直接拼接所有数据
            eeg_sample.extend(train_data_eeg)
            eye_sample.extend(train_data_eye)
            labels.extend(train_labels)

            eeg_sample.extend(test_data_eeg)
            eye_sample.extend(test_data_eye)
            labels.extend(test_labels)

        else:
            # 合并训练集和测试集
            eeg_data = np.concatenate((train_data_eeg, test_data_eeg), axis=0)
            eye_data = np.concatenate((train_data_eye, test_data_eye), axis=0)
            slabels = np.concatenate((train_labels, test_labels), axis=0)

            # 检查数据是否有相同的样本数
            assert eeg_data.shape[0] == eye_data.shape[0] == slabels.shape[0], "样本数不一致"

            # 随机打乱数据
            indices = np.random.permutation(eeg_data.shape[0])
            eeg_data = eeg_data[indices]
            eye_data = eye_data[indices]
            slabels = slabels[indices]

            # 划分support集和剩余集
            train_data_eeg, temp_data_eeg, train_data_eye, temp_data_eye, train_labels, temp_labels = manual_split(
                eeg_data, eye_data, slabels,
                num_support_per_class=parser.num_support_src,
                classes_per_it=parser.classes_per_it_src
            )

            # 从剩余集中划分验证集(300个)和测试集
            val_data_eeg, test_data_eeg, val_data_eye, test_data_eye, val_labels, test_labels = train_test_split(
                temp_data_eeg, temp_data_eye, temp_labels,
                train_size=300, random_state=42,  # 固定300个验证样本
                stratify=temp_labels)  # 分层采样，保持类别比例

            # 根据mode返回对应部分
            if mode == 'train':
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

"""
1. mode = 'full' （源域全部数据）
应用场景：提取那 7 个源域受试者的数据。
输出内容：这 7 个人的所有样本，不加任何切分。
输出形状：
EEG: (约 5838, 5, 9, 9) (7人 × 834)
Eye: (约 5838, 33)
Label: (约 5838,)
2. mode = 'train' （目标域 Support 集）
应用场景：提取第 8 个目标受试者的极少量参考样本（Support）。
输出内容：从这个人里面，每个情绪类别强制只抽出 1 个样本（3类 × 1个 = 3个）。
输出形状：
EEG: (3, 5, 9, 9)
Eye: (3, 33)
Label: (3,)
3. mode = 'val' （目标域验证集）
应用场景：提取第 8 个目标受试者的验证样本（用于 Early Stopping 防止过拟合）。
输出内容：代码里写死了 train_size=300，强制抽出 300 个样本。
输出形状：
EEG: (300, 5, 9, 9)
Eye: (300, 33)
Label: (300,)
4. mode = 'test' （目标域 Query 集 / 测试集）
应用场景：提取第 8 个目标受试者的待测样本（最终考试题）。
输出内容：这个人剩下的所有数据（834 - 3 - 300 = 531 个）。
输出形状：
EEG: (约 531, 5, 9, 9)
Eye: (约 531, 33)
Label: (约 531,)
"""

def reshape_feature(fts_sample):
    """
    重塑特征形状（可能用于某些处理）
    """
    fts = None
    if torch.is_tensor(fts_sample):
        elec, number, freq = fts_sample.shape
        fts = fts_sample.permute(1, 0, 2).reshape(number, elec * freq)
    else:
        print("Something Wrong!!!")

    return fts


def stack_bands(data_eeg):
    """
    将5个频带堆叠在一起（连接在特征维度）
    
    Args:
        data_eeg: 包含5个频带的字典，每个频带形状 (samples, 62)
    
    Returns:
        all_bands: 拼接后的数据，形状 (samples, 62*5)
    """
    all_bands = []
    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        all_bands.append(data_eeg[bands])  # 每个频段形状为 (499, 62)

    # 在特征维度上拼接
    all_bands = np.concatenate(all_bands, axis=1)  # 最终形状为 (499, 62 * 5)
    return all_bands


def exact_bands(data_eeg):
    """
    提取5个频带并在新维度上堆叠
    
    Args:
        data_eeg: 包含5个频带的字典，每个频带形状 (samples, 62)
    
    Returns:
        all_bands: 堆叠后的数据，形状 (samples, 62, 5)
    """
    all_bands = []
    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        all_bands.append(data_eeg[bands])  # 每个频段形状为 (499, 62)

    # 在第2个维度（通道维度）上堆叠
    all_bands = np.stack(all_bands, axis=2)  # 最终形状为 (499, 62, 5)
    return all_bands


def standardize_bands(data_eeg):
    """
    对每个频带分别进行标准化（Z-score）
    
    Args:
        data_eeg: 包含5个频带的字典，每个频带形状 (samples, 62)
    
    Returns:
        standardized_bands: 标准化后的字典
    """
    standardized_bands = {}

    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        scaler = StandardScaler()  # 减去均值，除以标准差
        # 对每个频带的每个通道进行标准化
        standardized_bands[band] = scaler.fit_transform(data_eeg[band])

    return standardized_bands


def data_1Dto2D_62chl(data, Y=9, X=9):
    """
    将62个通道的1D数据映射到9×9的2D拓扑图
    
    功能：按照电极在头皮的物理位置，将62个电极映射到9×9网格
         空白位置填充0
    
    Args:
        data: 62维的1D数组，形状 (62,)
        Y, X: 输出网格尺寸，固定9×9
    
    Returns:
        data_2D: 9×9的2D矩阵
    """
    data_2D = np.zeros([Y, X])
    
    # 按照电极位置映射（根据SEED数据集的电极布局）
    # 每一行对应9×9网格中的一行
    data_2D[0] = (0, 0, 0, data[0], data[1], data[2], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[3], 0, data[4], 0, 0, 0)
    data_2D[2] = (data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13])
    data_2D[3] = (data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22])
    data_2D[4] = (data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31])
    data_2D[5] = (data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39], data[40])
    data_2D[6] = (data[41], data[42], data[43], data[44], data[45], data[46], data[47], data[48], data[49])
    data_2D[7] = (0, data[50], data[51], data[52], data[53], data[54], data[55], data[56], 0)
    data_2D[8] = (0, 0, data[57], data[58], data[59], data[60], data[61], 0, 0)
    
    return data_2D


def convert_chl(data):
    """
    将62通道的1D数据转换为9×9的2D拓扑图
    
    Args:
        data: 输入数据，形状 (num_samples, 62, 5)
              样本数 × 62通道 × 5频带
    
    Returns:
        data_2d_all: 转换后的数据，形状 (num_samples, 5, 9, 9)
                     样本数 × 频带 × 9 × 9
    """
    num_samples, num_channels, num_bands = data.shape
    data_2d_all = np.zeros((num_samples, num_bands, 9, 9))
    
    # 对每个样本、每个频带进行转换
    for i in range(num_samples):
        for j in range(num_bands):
            data_1d = data[i, :, j]  # 62维向量
            data_2d = data_1Dto2D_62chl(data_1d)  # 转9×9
            data_2d_all[i, j, :, :] = data_2d
            
    return data_2d_all  # (num_samples, num_bands, 9, 9)


def load4data(parser, eeg_path, eye_path, eeg_session_names, eye_session_names, sess_idx, idx, mode):
    """
    加载SEED数据的顶层函数
    
    Args:
        parser: 配置参数
        eeg_path: EEG数据路径
        eye_path: 眼动数据路径
        eeg_session_names: EEG会话文件名列表
        eye_session_names: 眼动会话文件名列表
        sess_idx: 会话索引（1,2,3）
        idx: 被试索引列表
        mode: 'train'/'val'/'test'/'full'
    
    Returns:
        eeg_sample: EEG张量，形状 (样本数, 5, 9, 9)
        eye_sample: 眼动张量，形状 (样本数, 33)
        labels: 标签张量，形状 (样本数,)
    """
    eeg_sample, eye_sample, labels = load_data(parser, eeg_path, eye_path, eeg_session_names, eye_session_names, sess_idx, idx, mode)
    
    # 转换为PyTorch张量
    eeg_sample = np.array(eeg_sample)
    eeg_sample = torch.from_numpy(eeg_sample).type(torch.FloatTensor)

    eye_sample = np.array(eye_sample)
    eye_sample = torch.from_numpy(eye_sample).type(torch.FloatTensor)

    labels = np.array(labels)
    labels = torch.from_numpy(labels).type(torch.LongTensor)

    return eeg_sample, eye_sample, labels


"""
=============================================================================
数据预处理流程总结
=============================================================================

原始数据 -> load4data -> load_data -> 预处理 -> 返回张量

EEG预处理流程：
--------------------------------------------------------------------------------
1. 原始EEG: 包含5个频带的字典，每个频带 (samples, 62)
   │
   ├── standardize_bands()  # 每个频带分别Z-score标准化
   │    ↓
   ├── exact_bands()        # 堆叠频带: (samples, 62, 5)
   │    ↓
   └── convert_chl()         # 62通道映射到9×9网格
        ↓
  最终输出: (samples, 5, 9, 9)  [5个频带, 9×9空间拓扑]

眼动预处理流程：
--------------------------------------------------------------------------------
1. 原始眼动: 33个眼动特征（瞳孔直径、眨眼持续时间等）
   │
   └── MinMaxScaler()        # 归一化到[0,1]
        ↓
  最终输出: (samples, 33)

数据划分（Few-Shot设置）：
--------------------------------------------------------------------------------
对于每个被试：
- 原始数据被随机打乱
- 每个类别取出K个样本作为support集（论文中K=1,5,10,20）
- 剩余样本中，300个作为验证集，其余作为测试集

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `manual_split` | 2.D节 support set | 从每个类抽取K个support样本 |
| `standardize_bands` | 3.A节 preprocessing | EEG频带标准化 |
| `exact_bands` | 3.A节 | 提取5个频带特征 |
| `convert_chl` | 3.A节 + 文献[34] | 62通道→9×9拓扑图 |
| `data_1Dto2D_62chl` | 文献[34] | 电极位置映射 |
| `load4data` | 3.A节 | 整体数据加载接口 |

=============================================================================
关键点解释
=============================================================================

1. **为什么要把62通道映射到9×9网格？**
   - 保持电极的空间拓扑关系
   - 相邻电极在9×9网格中也相邻
   - 便于使用2D CNN提取空间特征

2. **为什么处理成5个频带？**
   - EEG有5个生理频带：delta, theta, alpha, beta, gamma
   - 不同频带反映不同的脑活动状态
   - 作为5个通道输入网络

3. **眼动数据为什么是33维？**
   - 论文3.A节提到：33个特征
   - 包括瞳孔直径、眨眼持续时间、注视点数量等统计特征

4. **manual_split的作用？**
   - 模拟Few-Shot学习的episode采样
   - 每个episode从每个类随机选K个support样本
   - 剩下的作为query（或进一步划分验证/测试）

5. **为什么要分train/val/test三种mode？**
   - train: support集，用于计算原型
   - val: 验证集，用于早停和调参
   - test: 测试集，用于最终评估
"""