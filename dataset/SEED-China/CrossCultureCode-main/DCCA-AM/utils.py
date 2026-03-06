import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle

from sklearn.preprocessing import StandardScaler


def LabelandNumber(eeg_data):
    train_data_eeg = pickle.loads( eeg_data['train_data'] )
    test_data_eeg = pickle.loads( eeg_data['test_data'] )
    train_label = eeg_data['train_label']
    test_label = eeg_data['test_label']

    train_data_all_bands = []
    test_data_all_bands = []

    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        train_tmp = train_data_eeg[bands]
        test_tmp = test_data_eeg[bands]
        if bands == 'delta':
            train_data_all_bands = train_tmp
            test_data_all_bands = test_tmp
        else:
            train_data_all_bands = np.hstack((train_data_all_bands, train_tmp))
            test_data_all_bands = np.hstack((test_data_all_bands, test_tmp))

    return train_data_all_bands.shape[0], test_data_all_bands.shape[0], train_label, test_label

def _stack_bands(data_eeg):
    all_bands = []
    for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        all_bands.append(data_eeg[bands])  # 每个频段形状为 (499, 62)

    # 使用 np.stack 在新轴上拼接频段，axis=2 表示在第2个维度上拼接
    all_bands = np.stack(all_bands, axis=2)  # 最终形状为 (499, 62, 5)
    return all_bands

def _standardize_bands(data_eeg):
    standardized_bands = {}

    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        scaler = StandardScaler()
        # Apply StandardScaler to each band's data across the 2D array dimensions
        standardized_bands[band] = scaler.fit_transform(data_eeg[band])

    return standardized_bands

def data_1Dto2D_62chl(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          data[0],    data[1],    data[2], 	0,  	    0, 	        0       )
    data_2D[1] = (0,  	   	0,          0,          data[3],    0,          data[4],    0,          0,          0       )
    data_2D[2] = (data[5],  data[6],    data[7],    data[8],    data[9],    data[10],   data[11],   data[12],   data[13])
    data_2D[3] = (data[14], data[15],   data[16],   data[17],   data[18],   data[19],   data[20],   data[21],   data[22])
    data_2D[4] = (data[23], data[24],   data[25],   data[26],   data[27],   data[28],   data[29],   data[30],   data[31])
    data_2D[5] = (data[32], data[33],   data[34],   data[35],   data[36],   data[37],   data[38],   data[39],   data[40])
    data_2D[6] = (data[41], data[42],   data[43],   data[44],   data[45],   data[46],   data[47],   data[48],   data[49])
    data_2D[7] = (0,        data[50],   data[51],   data[52],   data[53],   data[54],   data[55],   data[56],   0       )
    data_2D[8] = (0,        0,          data[57],   data[58],   data[59],   data[60],   data[61],   0,          0       )
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
def concat_process(eeg_data, eye_data):
    train_data_eye = np.asarray(eye_data['train_data_eye'].tolist()).squeeze()
    test_data_eye = np.asarray( eye_data['test_data_eye'].tolist()).squeeze()

    train_data_eeg = pickle.loads( eeg_data['train_data'] )
    test_data_eeg = pickle.loads( eeg_data['test_data'] )
    train_label = eeg_data['train_label']
    test_label = eeg_data['test_label']

    train_data_all_bands = []
    test_data_all_bands = []

    train_data_eeg = _standardize_bands(train_data_eeg)
    train_data_eeg = _stack_bands(train_data_eeg)

    # 1d channel vector to 2d channel matrix
    train_data_eeg = convert_chl(train_data_eeg)  # (12 * 499, 5, 9, 9)
    train_data_eeg = train_data_eeg.astype('float32')

    test_data_eeg = _standardize_bands(test_data_eeg)
    test_data_eeg = _stack_bands(test_data_eeg)

    test_data_eeg = convert_chl(test_data_eeg)  # (12 * 499, 5, 9, 9)
    test_data_eeg = test_data_eeg.astype('float32')

    # for bands in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    #     train_tmp = train_data_eeg[bands]
    #     test_tmp = test_data_eeg[bands]
    #     if bands == 'delta':
    #         train_data_all_bands = train_tmp
    #         test_data_all_bands = test_tmp
    #     else:
    #         train_data_all_bands = np.hstack((train_data_all_bands, train_tmp))
    #         test_data_all_bands = np.hstack((test_data_all_bands, test_tmp))

    # all_train = np.hstack((train_data_all_bands, train_data_eye))
    # all_test = np.hstack((test_data_all_bands, test_data_eye))

    return train_data_eeg, test_data_eeg, train_data_eye, test_data_eye, train_label, test_label


def generating_data(data_dict, clip_label, feature_name):
    # first 9 as training, the last 6 as testing
    train_data = data_dict[feature_name+'1']
    _, num, _ = train_data.shape
    train_label = np.zeros(num,) + clip_label[0]
    train_data = np.swapaxes(train_data, 0, 1)
    train_data = np.reshape(train_data, (num, -1))
    train_residual_index = [2,3,4,5,6,7,8,9]
    for ind,i in enumerate(train_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape
        used_label = np.zeros(num,) + clip_label[ind+1]
        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        train_data = np.vstack((train_data, used_data))
        train_label = np.hstack((train_label, used_label))

    test_data = data_dict[feature_name+'10']
    _, num, _ = test_data.shape
    test_label = np.zeros(num,) + clip_label[9]
    test_data = np.swapaxes(test_data, 0, 1)
    test_data = np.reshape(test_data, (num, -1))
    test_residual_index = [11,12,13,14,15]
    for ind,i in enumerate(test_residual_index):
        used_data = data_dict[feature_name + str(i)]
        _, num, _ = used_data.shape
        used_label = np.zeros(num,) + clip_label[ind+13]
        used_data = np.swapaxes(used_data, 0, 1)
        used_data = np.reshape(used_data, (num, -1))
        test_data = np.vstack((test_data, used_data))
        test_label = np.hstack((test_label, used_label))
    return train_data, test_data, train_label, test_label

def logistic_classification(train_data, test_data, train_label, test_label):
    best_res = {}
    best_res['n'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label
    best_res['proba'] = 0
    clf = LogisticRegression()
    clf.fit(train_data, train_label)
    p_labels = clf.predict(test_data)
    score = clf.score(test_data, test_label)
    proba = clf.predict_proba(test_data)
    best_res['acc'] = score
    best_res['p_label'] = p_labels
    best_res['proba'] = proba
    return best_res

def knn_classification(train_data, test_data, train_label, test_label):
    best_res = {}
    best_res['n'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label
    best_res['proba'] = 0
    for n in range(3, 10):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        proba = clf.predict_proba(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score
            best_res['n'] = n
            best_res['p_label'] = p_labels
            best_res['proba'] = proba
    return best_res

def svm_classification(train_data, test_data, train_label, test_label):
    best_res = {}
    best_res['c'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label
    for c in range(-10, 10):
        clf = svm.LinearSVC(C=2**c)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        decision_value = clf.decision_function(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score
            best_res['c'] = 2**c
            best_res['p_label'] = p_labels
            best_res['decision_val'] = decision_value
    for c in np.arange(0.1, 20, 0.5):
        clf = svm.LinearSVC(C=c)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        decision_value = clf.decision_function(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score
            best_res['c'] = c
            best_res['p_label'] = p_labels
            best_res['decision_val'] = decision_value
    return best_res
