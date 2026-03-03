"""
HIA-Net 训练脚本 (SEED数据集)
==================================================
功能：使用Hierarchical Interactive Alignment Network进行多模态小样本情感识别
数据集：SEED (12被试 x 3 sessions)

核心概念说明：
- LOSO (Leave-One-Subject-Out): 留一被试交叉验证，每次将一个被试作为测试集，其余被试用于训练
- N-way K-shot: 小样本学习设置，每轮迭代随机选择N个情感类别，每个类别提供K个支持样本
- Source vs Target: 源域（其他11个被试的数据）用于模型训练，目标域（当前被试数据）用于验证和测试
- ProtoNet: 原型网络，通过计算查询样本与各个类别原型（支持样本的均值）的距离进行分类
- GDD: Geodesic Distance Discrepancy，一种域适应损失，用于对齐源域和目标域的特征分布
"""

# ==================== 标准库导入 ====================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# ==================== PyTorch相关导入 ====================
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

# ==================== 自定义模块导入 ====================
from data_prepare.load_data import *
from data_prepare.prototypical_batch_sampler import *
from network import rescnn
from network.Cross_Att import *
from network.proto_att import *
from parse.parser_camp import get_parser
from utils.gdd import gdd
from utils.prototypical_loss import *
from utils.early_stop import *
from network.model import *


def set_seed(seed=42):
    """
    设置所有随机种子以确保实验可重复性。
    
    在深度学习中，随机性来自多个来源：PyTorch的随机数生成器、CUDA的随机数生成器等。
    此函数统一设置所有相关的随机种子，确保每次运行结果一致。
    
    参数:
        seed (int): 随机种子值，默认为42。使用固定数值可以保证可重复性。
    
    返回:
        None
    """
    torch.manual_seed(seed)           # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)       # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)   # 设置所有GPU的随机种子（如果是多卡训练）


def init_optim(opt, model, epoch=None):
    """
    初始化Adam优化器。
    
    Adam优化器结合了动量法和自适应学习率，是目前深度学习中最常用的优化器之一。
    
    参数:
        opt (argparse.Namespace): 命令行参数对象，包含learning_rate等配置
        model (nn.Module): 需要优化的神经网络模型
        epoch (int, optional): 当前训练轮次，预留参数，当前未使用
    
    返回:
        torch.optim.Adam: 配置好的Adam优化器实例
    """
    optim = torch.optim.Adam(
        params=model.parameters(),           # 需要优化的模型参数
        lr=opt.learning_rate,                 # 学习率，控制参数更新步长
        weight_decay=0.01                      # L2正则化系数，防止过拟合
    )
    return optim


def val(parser, data_dset, data_sampler, model, protonet):
    """
    在验证集上评估模型性能。
    
    验证阶段的主要目的是监控模型在未见过的目标域数据上的表现，
    用于早停判断和模型选择。验证过程使用目标域的训练集部分作为支持集，
    验证集部分作为查询集，模拟小样本学习场景。
    
    参数:
        parser (argparse.Namespace): 解析后的命令行参数
        data_dset (dict): 包含所有数据集的字典，键包括：
            - train_eeg_dset: 目标域训练集EEG数据
            - train_eye_dset: 目标域训练集眼动数据
            - val_eeg_dset: 目标域验证集EEG数据
            - val_eye_dset: 目标域验证集眼动数据
        data_sampler (dict): 包含所有采样器的字典
        model (nn.Module): 多模态融合模型
        protonet (ProtoNet): 原型网络，用于计算原型和距离
    
    返回:
        tuple: (val_proto_loss, val_proto_acc)
            - val_proto_loss (float): 验证集上的平均原型损失
            - val_proto_acc (float): 验证集上的平均分类准确率
    """
    # 确定运行设备（GPU或CPU）
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    
    # 设置为评估模式，这会关闭dropout和batch normalization的训练行为
    model.eval()
    
    # 初始化验证指标累加器
    val_proto_loss = 0.0
    val_proto_acc = 0.0

    # 获取训练集（源域）的迭代器，用于提供支持样本
    train_iter = iter(data_sampler["train_sampler"])
    train_eeg_dset = data_dset["train_eeg_dset"]
    train_eye_dset = data_dset["train_eye_dset"]

    # 获取验证集（目标域）的迭代器，用于提供查询样本
    val_iter = iter(data_sampler["val_sampler"])
    val_eeg_dset = data_dset["val_eeg_dset"]
    val_eye_dset = data_dset["val_eye_dset"]
    
    # 使用torch.no_grad()上下文管理器禁用梯度计算，减少内存消耗并加速计算
    with torch.no_grad():
        # 按照指定的迭代次数循环
        for it in range(parser.iterations):
            # ----- 获取支持集数据（来自训练集） -----
            source_indices = next(train_iter)  # 获取当前batch的索引
            source_eeg_batch, source_labels = train_eeg_dset[source_indices]  # 加载EEG数据和标签
            source_eye_batch, _ = train_eye_dset[source_indices]  # 加载对应的眼动数据（忽略标签）
            
            # ----- 获取查询集数据（来自验证集） -----
            val_indices = next(val_iter)
            val_eeg_batch, val_labels = val_eeg_dset[val_indices]
            val_eye_batch, _ = val_eye_dset[val_indices]

            # 将所有数据移动到指定设备（GPU/CPU）
            source_eeg_batch = source_eeg_batch.to(device)
            source_eye_batch = source_eye_batch.to(device)
            source_labels = source_labels.to(device)

            val_eeg_batch = val_eeg_batch.to(device)
            val_eye_batch = val_eye_batch.to(device)
            val_labels = val_labels.to(device)

            # ----- 前向传播：提取多模态融合特征 -----
            # 模型返回的是多层特征的列表，[-1]表示取最后一层（最高层）特征
            source_fusion = model(source_eeg_batch, source_eye_batch)
            val_fusion = model(val_eeg_batch, val_eye_batch)

            # ----- 计算原型网络的分类距离 -----
            # protonet接收支持集和查询集的特征，计算每个查询样本到各类原型的距离
            dist_tgt = protonet(
                [source_fusion[-1], val_fusion[-1]],  # 支持集和查询集的特征
                [source_labels, val_labels],          # 对应的标签
                parser.classes_per_it_tgt,             # 目标域的类别数
                parser.num_support_tgt,                 # 支持样本数量
                parser.num_query_tgt,                    # 查询样本数量
                flag=1                                   # 标志位，指示使用目标域设置
            )

            # ----- 计算原型损失和准确率 -----
            proto_loss, proto_acc, _ = prototypical_loss2(
                dist_tgt,                               # 距离矩阵
                parser.classes_per_it_tgt,               # 类别数
                parser.num_query_tgt,                     # 查询样本数
                parser                                    # 参数对象
            )

            # 累加指标并求平均（除以迭代次数）
            val_proto_acc += proto_acc / parser.iterations
            val_proto_loss += proto_loss / parser.iterations

    return val_proto_loss, val_proto_acc


def test(parser, data_dset, data_sampler, model, protonet, save_path):
    """
    在测试集上评估模型性能。
    
    测试阶段用于获得模型的最终性能指标。与验证阶段不同，测试集的数据在训练过程中
    完全未见过的。此函数还支持保存t-SNE可视化图，用于观察特征分布的对齐情况。
    
    参数:
        parser (argparse.Namespace): 命令行参数
        data_dset (dict): 数据集字典
        data_sampler (dict): 采样器字典
        model (nn.Module): 训练好的模型
        protonet (ProtoNet): 原型网络
        save_path (str): t-SNE可视化图的保存路径
    
    返回:
        tuple: (test_proto_loss, test_proto_acc, conf_matrix)
            - test_proto_loss (float): 测试集上的平均损失
            - test_proto_acc (float): 测试集上的平均准确率
            - conf_matrix (np.ndarray): 混淆矩阵，用于分析各类别的分类性能
    """
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    model.eval()

    test_proto_loss = 0.0
    test_proto_acc = 0.0

    # 获取源域（训练集）的采样器和数据集，用于提供支持样本
    train_iter = iter(data_sampler["train_sampler"])
    train_eeg_dset = data_dset["train_eeg_dset"]
    train_eye_dset = data_dset["train_eye_dset"]

    # 获取目标域（测试集）的采样器和数据集，用于提供查询样本
    test_target_iter = iter(data_sampler["test_sampler"])
    test_eeg_dset = data_dset["test_eeg_dset"]
    test_eye_dset = data_dset["test_eye_dset"]

    with torch.no_grad():
        for it in range(parser.iterations):
            # ----- 加载支持集数据（源域） -----
            source_indices = next(train_iter)
            source_eeg_batch, source_labels = train_eeg_dset[source_indices]
            source_eye_batch, _ = train_eye_dset[source_indices]

            # ----- 加载查询集数据（目标域） -----
            test_indices = next(test_target_iter)
            test_eeg_batch, test_labels = test_eeg_dset[test_indices]
            test_eye_batch, _ = test_eye_dset[test_indices]

            # 数据迁移到指定设备
            source_eeg_batch = source_eeg_batch.to(device)
            source_eye_batch = source_eye_batch.to(device)
            source_labels = source_labels.to(device)

            test_eeg_batch = test_eeg_batch.to(device)
            test_eye_batch = test_eye_batch.to(device)
            test_labels = test_labels.to(device)

            # ----- 特征提取 -----
            source_fusion = model(source_eeg_batch, source_eye_batch)
            test_fusion = model(test_eeg_batch, test_eye_batch)

            # ----- 原型网络分类 -----
            dist_tgt = protonet(
                [source_fusion[-1], test_fusion[-1]],
                [source_labels, test_labels],
                parser.classes_per_it_tgt,
                parser.num_support_tgt,
                parser.num_query_tgt,
                flag=1
            )

            # ----- 计算损失和准确率，并获取混淆矩阵 -----
            proto_loss, proto_acc, conf_matrix = prototypical_loss2(
                dist_tgt,
                parser.classes_per_it_tgt,
                parser.num_query_tgt,
                parser
            )

            # 累加指标
            test_proto_acc += proto_acc / parser.iterations
            test_proto_loss += proto_loss / parser.iterations

    return test_proto_loss, test_proto_acc, conf_matrix


# 设计超参数
eeg_input_dim = 256      # EEG特征的输入维度
eye_input_dim = 177      # 眼动特征的输入维度  
output_dim = 256         # 融合特征的输出维度
emotion_categories = 3   # 情感类别数（消极、中性、积极）


def main(parser, data_dset, data_sampler, writer, early_stopping):
    """
    主训练函数，执行完整的训练、验证和测试流程。
    
    训练流程包含以下关键步骤：
    1. 初始化模型和优化器
    2. 对每个epoch：
       a. 使用源域数据训练模型，计算分类损失
       b. 计算GDD损失，对齐源域和目标域的特征分布
       c. 反向传播更新模型参数
       d. 在验证集上评估
       e. 检查早停条件
    3. 在测试集上评估最佳模型和最后一个模型
    
    参数:
        parser (argparse.Namespace): 命令行参数
        data_dset (dict): 包含所有数据集的字典
        data_sampler (dict): 包含所有采样器的字典
        writer (SummaryWriter): TensorBoard日志写入器
        early_stopping (EarlyStoppingAccuracy): 早停控制器
    
    返回:
        tuple: (best_test_loss, best_test_acc, last_test_loss, last_test_acc, conf_matrix)
            包含最佳模型和最后模型的测试指标，以及选择的混淆矩阵
    """
    # ----- 初始化设备和随机种子 -----
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    set_seed(parser.seed)

    # ----- 创建模型 -----
    # MyModel是多模态融合模型，同时处理EEG和眼动数据
    model = MyModel(eeg_input_dim, eye_input_dim, output_dim)
    protonet = ProtoNet()  # 原型网络，用于小样本分类

    # 将模型移动到GPU（如果可用）
    model = model.to(device)
    protonet = protonet.to(device)

    # ----- 初始化优化器 -----
    params = list(model.parameters())
    optim = torch.optim.Adam(
        params=params,
        lr=parser.learning_rate,
        weight_decay=0.01  # L2正则化，防止过拟合
    )

    # 初始化最佳准确率记录
    best_acc = 0.0
    
    print("----------开始训练模型----------")
    
    # ==================== 训练主循环 ====================
    for epoch in range(1, parser.epochs + 1):
        print('========= Epoch: {} ========='.format(epoch))
        
        # ----- 设置为训练模式 -----
        model.train()
        protonet.train()

        # ----- 初始化训练指标累加器 -----
        train_loss = 0           # 总损失（分类损失 + GDD损失）
        train_gdd_loss = 0       # GDD域适应损失
        train_proto_loss_src = 0 # 源域分类损失
        train_proto_loss_tgt = 0 # 目标域分类损失（预留）
        train_proto_acc_src = 0  # 源域分类准确率
        train_proto_acc_tgt = 0  # 目标域分类准确率（预留）

        # ----- 获取数据迭代器 -----
        train_source_iter = iter(data_sampler["source_sampler"])  # 源域采样器（其他11个被试）
        train_iter = iter(data_sampler["train_sampler"])          # 目标域训练集采样器
        source_eeg_dset = data_dset["source_eeg_dset"]            # 源域EEG数据集
        source_eye_dset = data_dset["source_eye_dset"]            # 源域眼动数据集
        train_eeg_dset = data_dset["train_eeg_dset"]              # 目标域训练EEG数据集
        train_eye_dset = data_dset["train_eye_dset"]              # 目标域训练眼动数据集

        # ----- 每个epoch内的迭代训练 -----
        for it in range(parser.iterations):
            # 1. 加载源域数据（用于计算分类损失）
            source_indices = next(train_source_iter)
            source_eeg_batch, source_labels = source_eeg_dset[source_indices]
            source_eye_batch, _ = source_eye_dset[source_indices]

            # 2. 加载目标域数据（用于计算域适应损失）
            target_indices = next(train_iter)
            target_eeg_batch, target_labels = train_eeg_dset[target_indices]
            target_eye_batch, _ = train_eye_dset[target_indices]

            # 将数据移动到GPU
            source_eeg_batch = source_eeg_batch.to(device)
            source_eye_batch = source_eye_batch.to(device)
            source_labels = source_labels.to(device)
            target_eeg_batch = target_eeg_batch.to(device)
            target_eye_batch = target_eye_batch.to(device)
            target_labels = target_labels.to(device)

            # 3. 前向传播：提取源域和目标域的多层特征
            source_fusion = model(source_eeg_batch, source_eye_batch)  # 返回5层特征
            target_fusion = model(target_eeg_batch, target_eye_batch)

            # ===== GDD损失计算（核心域适应部分） =====
            # 计算每一层特征的GDD损失，用于对齐源域和目标域的特征分布
            gdd_losses = [
                gdd(src, tgt)  # 计算两个特征分布之间的测地线距离
                for src, tgt in zip(source_fusion, target_fusion)
            ]  # 得到长度为5的列表 [gdd_loss1, gdd_loss2, ..., gdd_loss5]

            # 4. 计算源域的分类损失
            dist_src = protonet(
                source_fusion[-1],  # 使用最后一层特征
                source_labels,
                parser.classes_per_it_src,
                parser.num_support_src,
                parser.num_query_src
            )
            proto_loss_src, proto_acc_src, _ = prototypical_loss2(
                dist_src,
                parser.classes_per_it_src,
                parser.num_query_src,
                parser
            )

            # ===== GDD损失的动态加权 =====
            # gamma是一个随epoch动态变化的权重因子，控制GDD损失的重要性
            # 使用sigmoid-like曲线：前期gamma小（域适应权重低），后期gamma大（域适应权重高）
            gamma = 2 / (1 + np.exp(-10 * (epoch) / (parser.epochs))) - 1

            # 为不同层级的特征分配不同的权重
            # 深层特征（对应更高的索引）获得更高的权重，因为它们包含更抽象的语义信息
            a = 0.5  # 权重缩放因子
            b = 1    # 对数函数的系数
            
            # 计算各层权重的原始值：使用对数函数确保权重随层级递增但增速递减
            raw_weights = a * np.log(b * np.arange(1, len(gdd_losses) + 1) + 1)
            # 归一化权重，使所有权重之和为1
            weights = raw_weights / raw_weights.sum()
            
            # 5. 合并多层GDD损失
            gdd_loss = gamma * sum(w * loss for w, loss in zip(weights, gdd_losses))
            
            # 6. 总损失 = 分类损失 + 域适应损失
            loss = proto_loss_src + gdd_loss

            # 7. 反向传播和优化
            optim.zero_grad()  # 清除之前的梯度
            loss.backward()     # 反向传播计算梯度
            optim.step()        # 更新模型参数

            # 累加训练指标（用于后续打印和TensorBoard记录）
            train_loss += loss / parser.iterations
            train_gdd_loss += gdd_loss / parser.iterations
            train_proto_loss_src += proto_loss_src / parser.iterations
            train_proto_acc_src += proto_acc_src / parser.iterations

        # ===== 记录训练指标到TensorBoard =====
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('gdd_loss', train_gdd_loss, epoch)
        writer.add_scalar('proto_loss_src', train_proto_loss_src, epoch)
        writer.add_scalar('proto_acc_src', train_proto_acc_src, epoch)

        # 将Tensor转换为Python数值（用于打印）
        train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        train_gdd_loss = train_gdd_loss.item() if isinstance(train_gdd_loss, torch.Tensor) else train_gdd_loss
        train_proto_loss_src = train_proto_loss_src.item() if isinstance(train_proto_loss_src, torch.Tensor) else train_proto_loss_src
        train_proto_loss_tgt = train_proto_loss_tgt.item() if isinstance(train_proto_loss_tgt, torch.Tensor) else train_proto_loss_tgt
        train_proto_acc_src = train_proto_acc_src.item() if isinstance(train_proto_acc_src, torch.Tensor) else train_proto_acc_src

        # 打印训练指标
        print('Train Loss: {:.6f}\tmmd_loss:{:.6f}\tproto_loss_src:{:.6f}\tproto_loss_tgt:{:.6f}\tproto_acc_src:{:.6f}\tproto_acc_tgt:{:.6f}'.format(
            train_loss, train_gdd_loss, train_proto_loss_src, train_proto_loss_tgt, train_proto_acc_src, train_proto_acc_tgt))

        # ===== 验证阶段 =====
        val_proto_loss, val_proto_acc = val(parser, data_dset, data_sampler, model, protonet)

        # 更新最佳验证准确率
        if val_proto_acc > best_acc:
            best_acc = val_proto_acc

        # ===== 早停检查 =====
        # 如果验证准确率连续多个epoch没有提升，则触发早停
        early_stopping(val_proto_acc, model)
        if early_stopping.early_stop:
            print("触发早停，训练结束")
            break

        # 打印验证结果
        print("验证 - Epoch: %d, 准确率: %f, 最佳准确率: %f" % (epoch, val_proto_acc, best_acc))

        # 记录验证指标到TensorBoard
        writer.add_scalar("val/loss", val_proto_loss, epoch)
        writer.add_scalar("val/Accuracy", val_proto_acc, epoch)
        writer.add_scalar("val/Best_Acc", best_acc, epoch)

    # ==================== 测试阶段 ====================
    save_path1 = "tsne_SEED_best.png"
    # 加载验证准确率最高的模型进行评估
    model.load_state_dict(torch.load(early_stopping.best_model_path))
    best_test_loss, best_test_acc, best_conf_matrix = test(parser, data_dset, data_sampler, model, protonet, save_path1)

    save_path2 = "tsne_SEED_last.png"
    # 加载最后一个epoch的模型进行评估
    model.load_state_dict(torch.load(early_stopping.last_model_path))
    last_test_loss, last_test_acc, last_conf_matrix = test(parser, data_dset, data_sampler, model, protonet, save_path2)

    # ===== 选择更好的模型结果 =====
    # 比较两个模型的测试准确率，选择表现更好的
    if best_test_acc > last_test_acc:
        conf_matrix = best_conf_matrix
    else:
        conf_matrix = last_conf_matrix

    return best_test_loss, best_test_acc, last_test_loss, last_test_acc, conf_matrix


def visualize_confusion_matrix(conf_matrix, class_names, save_path="confusion_matrix_SEED.png"):
    """
    可视化混淆矩阵。
    
    混淆矩阵是评估分类模型性能的重要工具，它展示了每个类别被正确分类和错误分类的情况。
    此函数将混淆矩阵归一化为百分比形式，并使用热力图进行可视化。
    
    参数:
        conf_matrix (np.ndarray): 混淆矩阵，形状为 (n_classes, n_classes)
        class_names (list): 类别名称列表，如 ['消极', '中性', '积极']
        save_path (str): 图像保存路径
    
    返回:
        None
    """
    # 计算每行的总和，用于归一化
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # 将计数转换为百分比（每行总和为100%）
    conf_matrix_percentage = conf_matrix / row_sums * 100

    plt.figure(figsize=(8, 6))

    # 设置字体（使用系统中存在的字体）
    plt.rc('font', family='DejaVu Sans')

    # 绘制热力图
    sns.heatmap(
        conf_matrix_percentage,
        annot=True,                    # 在单元格中显示数值
        fmt='.2f',                      # 数值格式为保留两位小数
        cmap='Blues',                    # 使用蓝色色系
        xticklabels=class_names,         # x轴标签
        yticklabels=class_names,         # y轴标签
        annot_kws={"size": 20, "weight": "bold"},  # 单元格内文字的格式
    )

    # 设置坐标轴标签字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18, rotation=0)

    # 保存图片（高dpi保证清晰度）
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵已保存为 {save_path}")


def plot_tsne_2d(embeddings_source, embeddings_target, labels_source, labels_target, save_path="tsne_SEED.png"):
    """
    使用t-SNE进行特征降维可视化。
    
    t-SNE是一种非线性降维技术，特别适合将高维特征降至2D或3D空间进行可视化。
    此函数将源域和目标域的特征同时可视化，可以直观地观察两个域的特征分布是否对齐。
    
    参数:
        embeddings_source (torch.Tensor): 源域特征，形状 (n_source_samples, feature_dim)
        embeddings_target (torch.Tensor): 目标域特征，形状 (n_target_samples, feature_dim)
        labels_source (torch.Tensor): 源域样本的标签
        labels_target (torch.Tensor): 目标域样本的标签
        save_path (str): 图像保存路径
    
    返回:
        None
    """
    # 分离计算图，避免梯度影响
    embeddings_source = embeddings_source.clone().detach()
    embeddings_target = embeddings_target.clone().detach()

    # 合并特征和标签
    embeddings = torch.cat([embeddings_source, embeddings_target], dim=0)
    labels = torch.cat([labels_source, labels_target])

    # 使用t-SNE进行降维
    # perplexity=5: 对于小样本数据集，使用较小的困惑度
    # early_exaggeration=10: 控制早期聚类力度
    tsne = TSNE(
        n_components=2,
        perplexity=5,
        early_exaggeration=10,
        learning_rate='auto',
        max_iter=500,
        random_state=42,
        method='exact',  # 使用精确算法（适合小数据集）
        angle=0.3
    )
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

    # 定义可视化参数
    colors = ['purple', 'yellow']  # 源域和目标域的颜色
    shapes = ['o', 's', '^']        # 不同情感类别的形状（圆、方、三角）
    domain_labels = ['源域', '目标域']
    emotion_labels = ['消极', '中性', '积极']

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制每个样本点
    for i in range(embeddings_2d.shape[0]):
        # 判断是源域还是目标域，选择相应颜色
        domain_color = colors[1] if i >= len(embeddings_source) else colors[0]
        # 根据情感类别选择形状
        shape = shapes[labels[i].item()]
        ax.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            c=domain_color,
            marker=shape,
            s=60,
            edgecolors='k'
        )

    # 添加图例
    # 情感类别图例（形状）
    scatter_proxies = [
        plt.Line2D([0], [0], linestyle='none', marker=shapes[i], color='k',
                   markersize=10, label=emotion_labels[i])
        for i in range(3)
    ]
    legend1 = ax.legend(handles=scatter_proxies, title="情感类别", loc="upper right")
    ax.add_artist(legend1)

    # 域图例（颜色）
    color_proxies = [
        plt.Line2D([0], [0], linestyle='none', marker='o', color=colors[i],
                   markersize=10, label=domain_labels[i])
        for i in range(2)
    ]
    ax.legend(handles=color_proxies, title="数据域", loc="lower left")

    # 添加网格
    ax.grid(True)

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        print(f"t-SNE可视化图已保存到 {save_path}")
    plt.close()


if __name__ == "__main__":
    # ==================== 主程序入口 ====================
    # 解析命令行参数
    parser = get_parser().parse_args()

    # ==================== 数据集配置 ====================
    # EEG数据文件名列表（每个session对应一组文件）
    eeg_session_names = [
        ['1_1.npz', '2_1.npz', '3_1.npz', '4_1.npz', '5_1.npz',
         '8_1.npz', '9_1.npz', '10_1.npz', '11_1.npz', '12_1.npz',
         '13_1.npz', '14_1.npz'],  # Session 1
        ['1_2.npz', '2_2.npz', '3_2.npz', '4_2.npz', '5_2.npz',
         '8_2.npz', '9_2.npz', '10_2.npz', '11_2.npz', '12_2.npz',
         '13_2.npz', '14_2.npz'],  # Session 2
        ['1_3.npz', '2_3.npz', '3_3.npz', '4_3.npz', '5_3.npz',
         '8_3.npz', '9_3.npz', '10_3.npz', '11_3.npz', '12_3.npz',
         '13_3.npz', '14_3.npz']   # Session 3
    ]

    # 眼动数据文件夹名列表
    eye_session_names = [
        ['1_1', '2_1', '3_1', '4_1', '5_1',
         '8_1', '9_1', '10_1', '11_1', '12_1',
         '13_1', '14_1'],  # Session 1
        ['1_2', '2_2', '3_2', '4_2', '5_2',
         '8_2', '9_2', '10_2', '11_2', '12_2',
         '13_2', '14_2'],  # Session 2
        ['1_3', '2_3', '3_3', '4_3', '5_3',
         '8_3', '9_3', '10_3', '11_3', '12_3',
         '13_3', '14_3']   # Session 3
    ]

    # 数据路径配置（需要根据实际环境修改）
    eeg_path = "/disk2/home/yuankang.fu/Datasets/SEED-China/02-EEG-DE-feature/eeg_used_4s"
    eye_path = "/disk2/home/yuankang.fu/Datasets/SEED-China/04-Eye-tracking-feature/eye_tracking_feature"

    # ==================== 跨session实验 ====================
    session_acc = []  # 记录每个session的平均准确率
    total_conf_matrix = np.zeros((3, 3))  # 累积所有被试的混淆矩阵

    # 对每个session进行实验
    for session in range(0, 3):
        sum_acc = 0.0
        mean_acc = 0.0
        acc_list = []  # 记录当前session中每个被试的准确率
        
        # 清空GPU缓存，避免显存碎片
        torch.cuda.empty_cache()
        
        idx = 0
        print('%%%%%%%%%% Session: {} %%%%%%%%%%'.format(session))
        
        # ==================== LOSO (留一被试交叉验证) ====================
        for subject in range(0, 12):
            idx += 1
            # 源域：除当前被试外的所有其他被试
            src_idx = [i for i in range(12) if i != subject]
            # 目标域：当前被试
            tar_idx = [subject]
            
            print(f"源域被试索引: {src_idx}")
            print(f"目标域被试索引: {tar_idx}")
            print('%%%%%%%%%% 目标被试: {} %%%%%%%%%%'.format(subject))

            # ----- 加载数据 -----
            # 源域数据（用于训练）
            source_eeg_sample, source_eye_sample, source_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, src_idx, "full"
            )
            
            # 目标域数据（划分训练/验证/测试集）
            train_eeg_sample, train_eye_sample, train_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, tar_idx, "train"
            )
            val_eeg_sample, val_eye_sample, val_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, tar_idx, "val"
            )
            test_eeg_sample, test_eye_sample, test_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, tar_idx, "test"
            )

            # ----- 创建TensorDataset（将数据和标签封装成PyTorch数据集）-----
            source_eeg_dset = torch.utils.data.TensorDataset(source_eeg_sample, source_label)
            source_eye_dset = torch.utils.data.TensorDataset(source_eye_sample, source_label)

            train_eeg_dset = torch.utils.data.TensorDataset(train_eeg_sample, train_label)
            train_eye_dset = torch.utils.data.TensorDataset(train_eye_sample, train_label)

            val_eeg_dset = torch.utils.data.TensorDataset(val_eeg_sample, val_label)
            val_eye_dset = torch.utils.data.TensorDataset(val_eye_sample, val_label)

            test_eeg_dset = torch.utils