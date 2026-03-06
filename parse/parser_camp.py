# coding=utf-8
import os
import argparse
from datetime import datetime

# 生成当前时间字符串，用于创建唯一的实验文件夹
# 格式：例如 "Mar02_14-30-45" (月日_时-分-秒)
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

dataset_name = 'SEED'  # 数据集名称：SEED 或 SEED-FRA
model_name = 'CAMP'    # 模型名称（可能是项目内部代号）

# =============================================================================
# 默认超参数配置 - 对应论文3.B节 Experimental Setup
# =============================================================================
params = {
    # ----- 实验路径配置 -----
    'dataset_name': dataset_name,
    'experiment_root': '../result_sdafsl/' + dataset_name + '/' + model_name + '/' + current_time,  # 实验结果保存路径
    
    # ----- 训练基本配置 -----
    'n_epochs': 50,  # 训练轮数，论文中设置：early stopping over 50 rounds
    'learning_rate': 1e-4,  # 学习率，论文中使用Adam优化器，lr=1e-4
    'lr_scheduler_step': 10,   # 学习率调整步长（每10个epoch调整一次）
    'lr_scheduler_gamma': 0.5, # 学习率衰减因子（乘以0.5）
    
    # ----- Episode配置（Few-Shot Learning的核心）-----
    'n_episodes': 20,  # 每个epoch中的episode数量，论文中：each with 20 episodes
    
    # 源域（Source Domain）配置 - 对应论文公式中的D_s
    'n_classes_per_episode_src': 3,   # 每个episode中的类别数N，论文中N=3（Positive, Neutral, Negative）
    'n_supports_per_class_src': 1,    # 每个类的support样本数K，论文中K=1,5,10,20
    'n_querys_per_class_src': 20,      # 每个类的query样本数，论文中每个类20个查询样本
    
    # 目标域（Target Domain）配置 - 对应论文公式中的D_tl（标记子集）
    'n_classes_per_episode_tgt': 3,   # 目标域类别数，同样为3
    'n_supports_per_class_tgt': 1,    # 目标域support样本数K（用于计算MMD损失）
    'n_querys_per_class_tgt': 20,      # 目标域query样本数（用于测试）
    
    # ----- 随机种子和硬件配置 -----
    'manual_seed': 42,    # 随机种子，确保实验可重复
    'cuda': 1,            # CUDA设备ID，-1表示使用CPU
    
    # ----- 优化器和早停配置 -----
    'optim': 'Adam',      # 优化器，论文中使用Adam
    'patience': 15,       # 早停耐心值，验证集性能连续15轮不提升则停止训练
    
    # ----- 数据相关配置 -----
    'num_bands': 5,       # EEG频带数，SEED数据集使用5个频带（delta,theta,alpha,beta,gamma）
    'num_sources': 12,    # 源域数量，SEED数据集有12个被试（中文被试）
    # 'num_cal': 120,     # 校准样本数，被注释掉了
    
    # ----- 其他配置 -----
    'SDA': False          # 是否使用目标域CE损失（可能是Source Domain Adaptation的缩写）
}


def get_parser():
    """
    创建命令行参数解析器
    功能：允许通过命令行覆盖默认的超参数配置
    对应论文：实验时可以方便地修改K-shot设置、学习率等
    
    Returns:
        parser: 配置好的ArgumentParser对象
    """
    parser = argparse.ArgumentParser()
    
    # ============ 数据集配置 ============
    parser.add_argument('-dataset', '--dataset',
                        type=str,
                        help='dataset name',  # 数据集名称：SEED 或 SEED-FRA
                        default=params['dataset_name'])

    # ============ 源域（Source）Episode配置 ============
    parser.add_argument('-cSrc', '--classes_per_it_src',
                        type=int,
                        help='number of random classes per episode of source',  # 每个episode中的类别数N
                        default=params['n_classes_per_episode_src'])

    parser.add_argument('-nsSrc', '--num_support_src',
                        type=int,
                        help='number of samples per class to use as support of source',  # 每个类的support样本数K
                        default=params['n_supports_per_class_src'])

    parser.add_argument('-nqSrc', '--num_query_src',
                        type=int,
                        help='number of samples per class to use as query of source',  # 每个类的query样本数
                        default=params['n_querys_per_class_src'])

    # ============ 目标域（Target）Episode配置 ============
    parser.add_argument('-cTgt', '--classes_per_it_tgt',
                        type=int,
                        help='number of random classes per episode of target',  # 目标域类别数
                        default=params['n_classes_per_episode_tgt'])

    parser.add_argument('-nsTgt', '--num_support_tgt',
                        type=int,
                        help='number of samples per class to use as support of target',  # 目标域support数
                        default=params['n_supports_per_class_tgt'])

    parser.add_argument('-nqTgt', '--num_query_tgt',
                        type=int,
                        help='number of samples per class to use as query of target',  # 目标域query数
                        default=params['n_querys_per_class_tgt'])

    # ============ 硬件和随机性配置 ============
    parser.add_argument('--cuda',
                        type=int,
                        help='enables cuda',  # CUDA设备ID
                        default=params['cuda'])

    parser.add_argument('--seed',
                        type=int,
                        help='input for the manual seeds initializations',  # 随机种子
                        default=params['manual_seed'])
    
    # ============ 训练过程配置 ============
    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=20',  # 每个epoch的episode数
                        default=params['n_episodes'])

    parser.add_argument('-optim', '--optim',
                        type=str,
                        help='which optimizer to use',  # 优化器类型
                        default=params['optim'])
                        
    parser.add_argument('-patience', '--patience',
                        type=str,
                        help='patience for early stopping',  # 早停耐心值
                        default=params['patience'])

    # ============ 数据维度配置 ============
    parser.add_argument('-num_bands', '--num_bands',
                        type=int,
                        help='number of EEG bands',  # EEG频带数
                        default=params['num_bands'])
                        
    parser.add_argument('-num_sources', '--num_sources',
                        type=int,
                        help='number of source domains',  # 源域数量（被试数）
                        default=params['num_sources'])

    # ============ 实验路径配置 ============
    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',  # 实验结果保存路径
                        default=params['experiment_root'])
                        
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',  # 训练轮数
                        default=params['n_epochs'])

    # ============ 优化器参数配置 ============
    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',  # 学习率
                        default=params['learning_rate'])

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=10',  # 学习率调整步长
                        default=params['lr_scheduler_step'])

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',  # 学习率衰减因子
                        default=params['lr_scheduler_gamma'])
                        
    parser.add_argument('--SDA',
                        type=bool,
                        help='whether use target CE loss',  # 是否使用目标域CE损失
                        default=params['SDA'])

    # ============ 数据路径配置 ============
   # ============ 数据路径配置 ============
    parser.add_argument('--eeg_path',
                        type=str,
                        help='path to EEG data directory',
                        # 将下面这行修改为你 Windows 上的路径（注意前面的 r）
                        default=r'D:\Work Dir\HIA-Net\dataset\SEED-Franch\EEG-DE-features\eeg_used_4s')

    parser.add_argument('--eye_path',
                        type=str,
                        help='path to eye-tracking data directory',
                        # 将下面这行修改为你 Windows 上的路径（注意前面的 r）
                        default=r'D:\Work Dir\HIA-Net\dataset\SEED-Franch\Eye-tracking-features\eye_tracking_feature')

    return parser


"""
=============================================================================
代码与论文对应关系
=============================================================================

| 参数名 | 论文对应部分 | 说明 |
|-------|------------|------|
| `n_epochs=50` | 3.B节 "early stopping over 50 rounds" | 训练50轮，配合早停 |
| `n_episodes=20` | 3.B节 "each with 20 episodes" | 每个epoch包含20个episode |
| `n_classes_per_episode_*=3` | 表I | 3种情感类别 |
| `n_supports_per_class_*=K` | 3.B节 "K=1,5,10 or 20" | Few-Shot的K-shot设置 |
| `n_querys_per_class_*=20` | 3.B节 "query sets (each N×20)" | 每个类20个查询样本 |
| `learning_rate=1e-4` | 3.B节 "learning rate of 1×10⁻⁴" | Adam优化器的学习率 |
| `num_bands=5` | 2.B.1节 | 5个EEG频带 |
| `num_sources=12` | 3.A.1节 | SEED数据集12个中文被试 |

=============================================================================
关键概念解释
=============================================================================

1. **什么是Episode？**
   - Few-Shot Learning的训练单元
   - 每个episode模拟一个N-way K-shot任务：
     * 随机选择N个类别（这里N=3）
     * 每个类别选K个support样本（K=1,5,10,20）
     * 每个类别选Q个query样本（Q=20）
   - 对应论文公式中的support set和query set

2. **为什么要有源域和目标域分开配置？**
   - 源域（Source）：有大量标注数据的被试
   - 目标域（Target）：只有少量标注数据的新被试
   - 训练时：用源域构造episode学习
   - 测试时：用目标域的K个support样本预测其余样本

3. **学习率调度器的作用？**
   - StepLR: 每10个epoch将学习率乘以0.5
   - 目的：训练后期用小学习率精细调整
   - 对应论文中可能使用的学习率衰减策略

4. **早停（Patience）的作用？**
   - 如果验证集准确率连续15轮不提升，停止训练
   - 防止过拟合，节省训练时间
   - 论文中确实提到了使用early stopping

=============================================================================
使用示例
=============================================================================

```bash
# 使用默认配置（1-shot）
python train.py

# 修改为5-shot设置
python train.py -nsSrc 5 -nsTgt 5

# 修改学习率
python train.py -lr 0.001

# 在SEED-FRA数据集上训练
python train.py -dataset SEED-FRA

# 不使用CUDA（使用CPU）
python train.py --cuda -1
"""