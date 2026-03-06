import os 
import numpy as np
import scipy.io as sio 
import pickle 

eye_feature_dir = './eye_tracking_feature/'
file_list = os.listdir(eye_feature_dir)
file_list.sort()

for item in file_list:
    print('*'*50)
    print(item)
    tmp_file = os.path.join(eye_feature_dir, item)
    tmp_data = pickle.load(open(tmp_file, 'rb'))
    print(tmp_data.keys())
    print(tmp_data['train_data_eye'].shape)
    print(tmp_data['test_data_eye'].shape)
    print('*'*50+'\n')