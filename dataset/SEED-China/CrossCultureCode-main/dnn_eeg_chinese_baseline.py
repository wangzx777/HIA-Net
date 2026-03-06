import numpy as np
import os
import torch
import torch.nn as nn

from sklearn import preprocessing
import pickle
import scipy.io as sio
from torch.utils.data import DataLoader

from seed_chn_dataloader import SEEDChinaDataset
import time


class dnn(nn.Module):
    def __init__(self, in_num=310, h1=128, h2=64, h3=32, out_num=3):
        super(dnn, self).__init__()
        self.dnn_net = nn.Sequential(
            nn.Linear(in_num, h1),
            nn.ReLU(),
            #nn.Dropout(p=0.5),  # 第一次 dropout
            nn.Linear(h1, h2),
            nn.ReLU(),
            #nn.Dropout(p=0.5),  # 第二次 dropout
            nn.Linear(h2, h3),
            nn.ReLU(),
            #nn.Dropout(p=0.5),  # 第三次 dropout
            nn.Linear(h3, out_num)
        )

    def forward(self, x):
        return self.dnn_net(x)


def train_dnn(train_loader, test_loader, lr, device):
    model = dnn(310, 128, 64, 32, 3)
    model.to(device)

    epoch_num = 15000
    learning_rate = lr
    # batch_size = 50

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    myloss = nn.CrossEntropyLoss()

    best_test_res = {
        'acc': 0,
        'predict_label': None,
        'trur_label': None
    }

    for ep in range(epoch_num):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 遍历训练数据的每个批次
        for batch_idx, (train_data, train_label) in enumerate(train_loader):
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            # print(f"train_label.shape:{train_label.shape}")
            # 前向传播
            output = model(train_data)
            loss = myloss(output, train_label)
            train_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(output, 1)
            correct += (predicted == train_label).sum().item()
            total += train_label.size(0)

        train_acc = correct / total
        print(f'Epoch : {ep} -- TrainLoss : {train_loss / len(train_loader)} -- TrainAcc : {train_acc}')

        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for test_data, test_label in test_loader:
                test_data = test_data.to(device)
                test_label = test_label.to(device)

                test_output = model(test_data)
                loss = myloss(test_output, test_label)
                test_loss += loss.item()

                _, predicted = torch.max(test_output, 1)
                correct += (predicted == test_label).sum().item()
                total += test_label.size(0)

        test_acc = correct / total
        print(f'Epoch : {ep} -- TestLoss : {test_loss / len(test_loader)} -- TestAcc : {test_acc}')

        # 保存最好的测试结果
        if best_test_res['acc'] < test_acc:
            best_test_res['acc'] = test_acc
            best_test_res['predict_label'] = predicted.cpu().numpy()
            best_test_res['true_label'] = test_label.cpu().numpy()
            print('update res')

    return best_test_res


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'





res_dir = './11_dnn_eeg_1s_baseline/'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

learning_rate = [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003,
                 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09]

# dataloader 实例化
eeg_path = "D:\Research\ML\Few-shot\SEED-China\\02-EEG-DE-feature\eeg_used_4s"  # Update with actual path

eeg_session_names = ['1_1.npz', '2_1.npz', '3_1.npz', '4_1.npz', '5_1.npz',
                     '8_1.npz', '9_1.npz', '10_1.npz', '11_1.npz', '12_1.npz',
                     '13_1.npz', '14_1.npz',
                     '1_2.npz', '2_2.npz', '3_2.npz', '4_2.npz', '5_2.npz',
                     '8_2.npz', '9_2.npz', '10_2.npz', '11_2.npz', '12_2.npz',
                     '13_2.npz', '14_2.npz',
                     '1_3.npz', '2_3.npz', '3_3.npz', '4_3.npz', '5_3.npz',
                     '8_3.npz', '9_3.npz', '10_3.npz', '11_3.npz', '12_3.npz',
                     '13_3.npz', '14_3.npz']
train_dataset = SEEDChinaDataset(eeg_path=eeg_path, eeg_session_names=eeg_session_names, train=True)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
print(train_dataset.__len__())

test_dataset = SEEDChinaDataset(eeg_path=eeg_path, eeg_session_names=eeg_session_names, train=False)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
print(test_dataset.__len__())

train_dnn(train_loader, test_loader, 0.005, device)
# for idx, lr in enumerate(learning_rate):
#     best_res = train_dnn(train_loader, test_loader, lr, device)
#     if not os.path.exists(os.path.join(res_dir, str(idx))):
#         os.mkdir(os.path.join(res_dir, str(idx)))
#     pickle.dump(best_res, open(os.path.join(res_dir, str(idx)), 'wb'))
