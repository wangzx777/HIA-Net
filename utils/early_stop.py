import os
import numpy as np
import torch


class EarlyStoppingAccuracy:
    def __init__(self, patience=7, verbose=False, delta=0, path='models/', individual_id=None, session_id=None):
        """
        Args:
            patience (int): 在验证集准确率不提升的情况下，允许的最大 epoch 数。默认值: 7
            verbose (bool): 是否打印每次验证集准确率提升的信息。默认值: False
            delta (float): 定义提升的最小幅度。默认值: 0
            path (str): 模型保存的基础路径，默认保存到 'models/' 目录
            individual_id (str): 当前个体编号，用于动态生成路径
            session_id (str): 当前会话编号，用于动态生成路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc_max = -np.Inf
        self.delta = delta
        self.individual_id = individual_id
        self.session_id = session_id
        self.path = path
        self.best_model_path = self._generate_dynamic_path('best_model.pth')
        self.last_model_path = self._generate_dynamic_path('last_model.pth')

    def _generate_dynamic_path(self, filename):
        """根据当前个体和会话生成动态文件名路径"""
        dynamic_path = self.path
        if self.individual_id is not None and self.session_id is not None:
            dynamic_path = os.path.join(self.path, f'individual_{self.individual_id}_session_{self.session_id}_{filename}')
        elif self.individual_id is not None:
            dynamic_path = os.path.join(self.path, f'individual_{self.individual_id}_{filename}')
        elif self.session_id is not None:
            dynamic_path = os.path.join(self.path, f'session_{self.session_id}_{filename}')
        return dynamic_path

    def __call__(self, val_acc, model):
        """每个 epoch 后进行模型检查，保存 best_model 和 last_model"""
        score = val_acc

        # 每个 epoch 保存 last_model
        self.save_last_model(model)

        if self.best_score is None:
            self.best_score = score
            self.save_best_model(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best_model(val_acc, model)
            self.counter = 0

    def save_best_model(self, val_acc, model):
        """保存验证集准确率提升的 best_model"""
        # 确保保存路径存在
        if not os.path.exists('models'):
            os.makedirs('models')

        if self.verbose:
            print(f'Validation accuracy increased ({self.acc_max:.6f} --> {val_acc:.6f}). Saving best model to {self.best_model_path}...')
        torch.save(model.state_dict(), self.best_model_path)  # 保存 best_model
        self.acc_max = val_acc

    def save_last_model(self, model):
        """保存当前 epoch 的模型为 last_model"""
        # 确保保存路径存在
        if not os.path.exists('models'):
            os.makedirs('models')

        if self.verbose:
            print(f'Saving last model to {self.last_model_path}...')
        torch.save(model.state_dict(), self.last_model_path)  # 保存 last_model
