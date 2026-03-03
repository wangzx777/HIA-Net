import torch
import numpy as np

def find_class_by_name(name, modules):
    """
    在模块列表中搜索指定名称的类并返回
    
    Args:
        name: 要查找的类名
        modules: 模块列表，例如 [module1, module2, ...]
    
    Returns:
        找到的类
    
    用途：动态加载模型或类，根据字符串名称获取对应的类对象
    例如：find_class_by_name('ResCBAM', [models]) 返回 ResCBAM 类
    """
    # 从每个模块中获取名为name的属性，如果不存在则返回None
    modules = [getattr(module, name, None) for module in modules]
    
    # 返回第一个非None的模块
    # next() 从迭代器中获取第一个元素
    # 这里使用生成器表达式过滤掉None值
    return next(a for a in modules if a)


def to_cuda(x):
    """
    将张量移动到GPU（如果CUDA可用）
    
    Args:
        x: PyTorch张量
    
    Returns:
        如果CUDA可用，返回cuda()后的张量；否则返回原张量
    
    用途：简化设备迁移代码，自动判断是否使用GPU
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_data(x):
    """
    将张量转换为numpy数组（从GPU移动到CPU）
    
    Args:
        x: PyTorch张量
    
    Returns:
        numpy数组形式的张量数据
    
    用途：在训练结束后，将结果转换为numpy格式以便可视化或保存
    """
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def to_onehot(label, num_classes):
    """
    将标签索引转换为one-hot编码
    
    Args:
        label: 标签索引，形状 (batch_size,) 或标量
        num_classes: 类别总数
    
    Returns:
        onehot: one-hot编码，形状 (batch_size, num_classes)
    
    示例：
        label = [0, 2, 1], num_classes=3
        onehot = [
            [1, 0, 0],
            [0, 0, 1], 
            [0, 1, 0]
        ]
    """
    # 创建单位矩阵：每个类别对应一个one-hot向量
    identity = to_cuda(torch.eye(num_classes))
    
    # 根据标签索引选择对应的one-hot向量
    onehot = torch.index_select(identity, 0, label)
    return onehot


def mean_accuracy(preds, target):
    """
    计算平均类别准确率（每个类别的准确率取平均）
    
    Args:
        preds: 模型预测，形状 (batch_size, num_classes) 概率值
        target: 真实标签，形状 (batch_size,)
    
    Returns:
        mean_acc: 平均类别准确率（百分比）
    
    用途：当类别不平衡时，使用平均准确率比整体准确率更公平
         论文中可能使用这个指标评估每个情感类别的识别效果
    
    示例：
        preds = [[0.7,0.2,0.1], [0.2,0.6,0.2], [0.1,0.2,0.7]]
        target = [0, 1, 2]
        类别0准确率: 1/1 = 100%
        类别1准确率: 1/1 = 100%
        类别2准确率: 1/1 = 100%
        mean_acc = 100%
    """
    # 获取预测类别（概率最大的类）
    preds = torch.max(preds, dim=1).indices  # (batch_size,)
    
    num_classes = preds.size(1) if len(preds.shape) > 1 else len(torch.unique(target))
    accu_class = []
    
    # 对每个类别分别计算准确率
    for c in range(num_classes):
        # 找出真实标签为c的样本
        mask = (target == c)
        c_count = torch.sum(mask).item()
        
        # 如果该类没有样本，跳过
        if c_count == 0:
            continue
            
        # 在这些样本中，找出预测正确的数量
        preds_c = torch.masked_select(preds, mask)
        correct = torch.sum(preds_c == c).item()
        
        # 计算该类别的准确率
        accu_class.append(1.0 * correct / c_count)
    
    # 返回所有类别准确率的平均值
    return 100.0 * np.mean(accu_class)


def accuracy(preds, target):
    """
    计算整体准确率（所有样本的正确率）
    
    Args:
        preds: 模型预测，形状 (batch_size, num_classes) 概率值
        target: 真实标签，形状 (batch_size,)
    
    Returns:
        acc: 整体准确率（百分比）
    
    用途：基本的分类准确率指标
    
    示例：
        preds = [[0.7,0.2,0.1], [0.2,0.6,0.2], [0.1,0.3,0.6]]
        target = [0, 1, 2]
        预测类别: [0, 1, 2]
        正确数: 3
        acc = 100%
    """
    # 获取预测类别（概率最大的类）
    preds = torch.max(preds, dim=1).indices  # (batch_size,)
    
    # 计算正确预测的比例
    return 100.0 * torch.sum(preds == target).item() / preds.size(0)


"""
=============================================================================
函数用途总结
=============================================================================

| 函数名 | 用途 | 在HIA-Net中的应用场景 |
|-------|------|----------------------|
| `find_class_by_name` | 动态加载类 | 根据配置文件动态选择模型 |
| `to_cuda` | 张量移至GPU | 训练前将数据移到GPU |
| `to_data` | 张量转numpy | 保存结果、可视化 |
| `to_onehot` | 标签转one-hot | 可能用于某些损失函数 |
| `mean_accuracy` | 平均类别准确率 | 评估每个情感类别的性能 |
| `accuracy` | 整体准确率 | 表II中的性能指标 |

=============================================================================
使用示例
=============================================================================

```python
# 1. 动态加载模型
from models import rescnn, Cross_Att
ModelClass = find_class_by_name('ResCBAM', [rescnn])
model = ModelClass()

# 2. 数据迁移
x = torch.randn(64, 5, 9, 9)
x = to_cuda(x)  # 自动移到GPU

# 3. 计算准确率
preds = model(x)  # (64, 3)
target = torch.randint(0, 3, (64,))

acc = accuracy(preds, target)  # 整体准确率
mean_acc = mean_accuracy(preds, target)  # 平均类别准确率

# 4. one-hot编码
labels = torch.tensor([0, 2, 1])
onehot = to_onehot(labels, 3)
# onehot = [[1,0,0], [0,0,1], [0,1,0]]

# 5. 保存结果
numpy_array = to_data(preds)  # 转为numpy保存
"""