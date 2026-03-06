import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle

def generating_data(data_dict, clip_label, feature_name):
    # 获取数据字典中的数据
    feature_data = pickle.loads(data_dict['train_data'])
    print(feature_data.keys())
    # print(feature_data.shape)
    if feature_data is None:
        raise KeyError(f"Feature '{feature_name}' not found in the data dictionary.")


    # 按照之前的逻辑进行处理
    train_data = feature_data['delta']  # 假设 'delta' 是你想要的特征
    num, _ = train_data.shape
    train_label = np.zeros(num,) + clip_label[0]
    train_data = np.swapaxes(train_data, 0, 1)
    train_data = np.reshape(train_data, (num, -1))

    train_residual_index = [2, 3, 4, 5, 6, 7, 8, 9]
    for ind, i in enumerate(train_residual_index):
        used_data = feature_data['delta']  # 修改为适合的索引或特征名称
        num, _ = used_data.shape
        used_label = np.zeros(num,) + clip_label[ind + 1]
        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        train_data = np.vstack((train_data, used_data))
        train_label = np.hstack((train_label, used_label))


    # # 类似的处理 test_data
    # test_data = feature_data['delta']  # 修改为适合的索引或特征名称
    # _, num, _ = test_data.shape
    # test_label = np.zeros(num,) + clip_label[9]
    # test_data = np.swapaxes(test_data, 0, 1)
    # test_data = np.reshape(test_data, (num, -1))
    # test_residual_index = [11, 12, 13, 14, 15]
    # for ind, i in enumerate(test_residual_index):
    #     used_data = feature_data['delta']  # 修改为适合的索引或特征名称
    #     _, num, _ = used_data.shape
    #     used_label = np.zeros(num,) + clip_label[ind + 13]
    #     used_data = np.swapaxes(used_data, 0, 1)
    #     used_data = np.reshape(used_data, (num, -1))
    #     test_data = np.vstack((test_data, used_data))
    #     test_label = np.hstack((test_label, used_label))
    #
    # return train_data, test_data, train_label, test_label


