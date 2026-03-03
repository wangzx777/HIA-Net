
import os
import numpy as np
import torch


class EarlyStoppingAccuracy:
    """
    早停机制 - 对应论文3.B节的早停策略
    
    功能：监控验证集准确率，当准确率不再提升时提前终止训练
    论文原文："An early stopping scheme reduces training time over 50 rounds"
    
    主要作用：
    1. 防止过拟合
    2. 节省训练时间
    3. 保存最佳模型
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='models/', individual_id=None, session_id=None):
        """
        Args:
            patience (int): 在验证集准确率不提升的情况下，允许的最大 epoch 数。默认值: 7
                           论文中训练50轮，如果连续7轮不提升就停止
            verbose (bool): 是否打印每次验证集准确率提升的信息。默认值: False
            delta (float): 定义提升的最小幅度。默认值: 0
                          只有准确率提升超过delta才认为是有效提升
            path (str): 模型保存的基础路径，默认保存到 'models/' 目录
            individual_id (str): 当前个体编号，用于动态生成路径（被试ID）
            session_id (str): 当前会话编号，用于动态生成路径（实验会话）
        """
        self.patience = patience  # 早停耐心值
        self.verbose = verbose    # 是否打印详细信息
        self.counter = 0          # 连续不提升的epoch计数器
        self.best_score = None    # 当前最佳准确率
        self.early_stop = False   # 是否触发早停
        self.acc_max = -np.Inf    # 历史最高准确率（初始化为负无穷）
        self.delta = delta         # 最小提升阈值
        self.individual_id = individual_id  # 被试ID，用于跨被试实验
        self.session_id = session_id        # 会话ID，SEED数据集有3个session
        self.path = path                    # 模型保存路径
        
        # 生成动态的模型保存路径（包含被试和会话信息）
        self.best_model_path = self._generate_dynamic_path('best_model.pth')   # 最佳模型路径
        self.last_model_path = self._generate_dynamic_path('last_model.pth')   # 最后模型路径

    def _generate_dynamic_path(self, filename):
        """
        根据当前个体和会话生成动态文件名路径
        
        论文中的leave-one-subject-out交叉验证需要为每个被试保存单独的模型
        路径格式示例：
        - 有被试和会话: models/individual_5_session_1_best_model.pth
        - 只有被试: models/individual_5_best_model.pth
        - 只有会话: models/session_1_best_model.pth
        - 默认: models/best_model.pth
        """
        dynamic_path = self.path
        if self.individual_id is not None and self.session_id is not None:
            # 例如：models/individual_5_session_1_best_model.pth
            dynamic_path = os.path.join(self.path, f'individual_{self.individual_id}_session_{self.session_id}_{filename}')
        elif self.individual_id is not None:
            # 例如：models/individual_5_best_model.pth
            dynamic_path = os.path.join(self.path, f'individual_{self.individual_id}_{filename}')
        elif self.session_id is not None:
            # 例如：models/session_1_best_model.pth
            dynamic_path = os.path.join(self.path, f'session_{self.session_id}_{filename}')
        return dynamic_path

    def __call__(self, val_acc, model):
        """
        每个 epoch 后调用，检查是否需要早停并保存模型
        
        Args:
            val_acc: 当前epoch的验证集准确率
            model: 当前epoch的PyTorch模型
        
        工作流程：
        1. 先保存last_model（每个epoch都保存）
        2. 检查当前准确率是否是最佳的
           - 如果是：保存为best_model，重置计数器
           - 如果不是：计数器+1
        3. 如果连续patience轮都没有提升，触发早停
        """
        score = val_acc

        # 每个 epoch 保存 last_model（用于恢复训练或分析）
        self.save_last_model(model)

        if self.best_score is None:
            # 第一个epoch，直接保存为最佳模型
            self.best_score = score
            self.save_best_model(val_acc, model)
        elif score < self.best_score + self.delta:
            # 当前准确率没有超过最佳准确率 + 最小阈值
            # 说明没有显著提升
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            # 如果连续patience轮都没有提升，触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 当前准确率有显著提升，更新最佳模型
            self.best_score = score
            self.save_best_model(val_acc, model)
            self.counter = 0  # 重置计数器

    def save_best_model(self, val_acc, model):
        """
        保存验证集准确率提升的 best_model
        当发现更好的模型时调用
        
        Args:
            val_acc: 当前验证集准确率
            model: PyTorch模型
        """
        # 确保保存路径存在
        if not os.path.exists('models'):
            os.makedirs('models')

        if self.verbose:
            print(f'Validation accuracy increased ({self.acc_max:.6f} --> {val_acc:.6f}). Saving best model to {self.best_model_path}...')
        
        # 保存模型参数（state_dict），不保存整个模型对象
        torch.save(model.state_dict(), self.best_model_path)  # 保存 best_model
        self.acc_max = val_acc  # 更新历史最高准确率

    def save_last_model(self, model):
        """
        保存当前 epoch 的模型为 last_model
        每个epoch都会调用，用于保存训练过程中的所有中间模型
        
        Args:
            model: PyTorch模型
        """
        # 确保保存路径存在
        if not os.path.exists('models'):
            os.makedirs('models')

        if self.verbose:
            print(f'Saving last model to {self.last_model_path}...')
        
        # 保存模型参数
        torch.save(model.state_dict(), self.last_model_path)  # 保存 last_model


"""
=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `EarlyStoppingAccuracy` | 3.B节 "early stopping scheme" | 早停机制，减少训练时间 |
| `patience=7` | 未明确说明 | 连续7轮不提升则停止 |
| `individual_id` | 3.B节 "leave-one-subject-out" | 被试ID，用于跨被试验证 |
| `session_id` | 3.A节 "three sessions" | SEED数据集的3个实验会话 |
| `best_model_path` | - | 保存最佳模型用于测试 |
| `last_model_path` | - | 保存最后一轮模型用于分析 |

=============================================================================
数据流示例
=============================================================================

训练过程示例（假设patience=3）：
------------------------------------------------------------------------------
Epoch 1: val_acc=0.75
    - 保存 last_model (epoch1_model.pth)
    - best_score=None，保存为 best_model (acc=0.75)
    counter=0

Epoch 2: val_acc=0.76
    - 保存 last_model (epoch2_model.pth)
    - 0.76 > 0.75，有提升，更新 best_model (acc=0.76)
    counter=0

Epoch 3: val_acc=0.74
    - 保存 last_model (epoch3_model.pth)
    - 0.74 < 0.76，无提升，counter=1

Epoch 4: val_acc=0.75
    - 保存 last_model (epoch4_model.pth)
    - 0.75 < 0.76，无提升，counter=2

Epoch 5: val_acc=0.73
    - 保存 last_model (epoch5_model.pth)
    - 0.73 < 0.76，无提升，counter=3 >= patience
    - 触发早停，early_stop=True

训练提前结束于Epoch 5，最佳模型是Epoch 2的模型。

=============================================================================
关键点解释
=============================================================================

1. **为什么需要早停？**
   - 论文训练50轮，但不是每次都要跑满50轮
   - 当模型不再提升时，继续训练只会浪费时间和算力
   - 防止过拟合：训练太久可能记住训练集，泛化能力下降

2. **为什么要保存两种模型？**
   - best_model: 验证集上表现最好的模型，用于最终测试
   - last_model: 最后一轮的模型，可用于分析训练过程
   - 有些情况下last_model可能过拟合，不如best_model好

3. **为什么路径中包含individual_id和session_id？**
   - 论文使用leave-one-subject-out交叉验证
   - 每个被试轮流作为目标域，需要保存多个模型
   - SEED数据集有3个session，需要区分不同session的模型

4. **delta参数的作用？**
   - 定义"提升"的最小阈值
   - delta=0表示只要有提升就算
   - delta=0.01表示准确率必须提高1%以上才算提升
   - 避免微小波动导致的误判

=============================================================================
使用示例
=============================================================================

```python
# 在训练循环中的使用方式
def train_one_epoch(...):
    # ... 训练代码 ...
    return val_acc

# 初始化早停器
early_stopping = EarlyStoppingAccuracy(
    patience=7, 
    verbose=True, 
    path='./models',
    individual_id=5,  # 第5个被试作为目标域
    session_id=1      # 第1个session
)

for epoch in range(50):
    val_acc = train_one_epoch()
    
    # 检查早停
    early_stopping(val_acc, model)
    
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# 加载最佳模型进行测试
model.load_state_dict(torch.load(early_stopping.best_model_path))
test_acc = test(model)
```
"""

# 可能的扩展：添加加载最佳模型的方法
# def load_best_model(self, model):
#     """加载之前保存的最佳模型"""
#     if os.path.exists(self.best_model_path):
#         model.load_state_dict(torch.load(self.best_model_path))
#         print(f"Loaded best model from {self.best_model_path}")
#     return model

"""

这个早停机制是训练过程中非常重要的部分，它确保了：

1. **训练效率**：不需要跑满50轮，模型收敛后自动停止
2. **模型质量**：保存验证集上最好的模型，而不是最后一轮
3. **实验管理**：为每个被试、每个会话单独保存模型

"""