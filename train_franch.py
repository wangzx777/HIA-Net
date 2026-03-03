"""
HIA-Net 训练脚本 (SEED-Franch数据集)
==================================================
功能：使用Hierarchical Interactive Alignment Network进行多模态小样本情感识别
数据集：SEED-Franch (8被试 x 3 sessions)

与train.py的区别：
1. 数据集不同：SEED-Franch (8被试) vs SEED (12被试)
2. 数据路径不同：SEED-Franch/ vs SEED-China/
3. 文件名列表不同

核心概念说明：
- LOSO (Leave-One-Subject-Out): 留一被试交叉验证，每次将一个被试作为测试集，其余被试用于训练
- N-way K-shot: 小样本学习设置，每轮迭代随机选择N个情感类别，每个类别提供K个支持样本
- Source vs Target: 源域（其他7个被试的数据）用于模型训练，目标域（当前被试数据）用于验证和测试
- ProtoNet: 原型网络，通过计算查询样本与各个类别原型（支持样本的均值）的距离进行分类
- GDD: Geodesic Distance Discrepancy，一种域适应损失，用于对齐源域和目标域的特征分布
"""

# ==================== 标准库导入 ====================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== PyTorch相关导入 ====================
import torch
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
    
    在深度学习中，随机性来自多个来源：PyTorch的随机数生成器（用于权重初始化、dropout等）、
    CUDA的随机数生成器等。此函数统一设置所有相关的随机种子，确保每次运行结果一致。
    
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
        params=model.parameters(),     # 需要优化的模型参数
        lr=opt.learning_rate            # 学习率，控制参数更新步长
    )
    return optim


# =========================================================================
# 验证集测试阶段 (Validation)
# =========================================================================
def val(parser, data_dset, data_sampler, model, protonet):
    """
    在验证集上评估模型性能。
    
    验证阶段的主要目的是监控模型在未见过的目标域数据上的表现，
    用于早停判断和模型选择。验证过程使用目标域的训练集部分作为支持集，
    验证集部分作为查询集，模拟小样本学习场景。
    
    参数:
        parser (argparse.Namespace): 解析后的命令行参数
        data_dset (dict): 包含所有数据集的字典
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
    
    # 【PyTorch机制】设置为评估模式，这会冻结BatchNorm的均值和方差，并关闭Dropout，
    # 保证测试时不受随机性影响，得到确定性的结果。
    model.eval()
    
    # 初始化验证指标累加器
    val_proto_loss = 0.0
    val_proto_acc = 0.0

    # 【重要提醒：变量命名陷阱】
    # 这里的 data_sampler["train_sampler"] 实际上包含的是【目标域（当前测试者）的支持集样本】
    # 这些样本作为"参考答案"，用于构建类别原型
    train_iter = iter(data_sampler["train_sampler"])
    train_eeg_dset = data_dset["train_eeg_dset"]
    train_eye_dset = data_dset["train_eye_dset"]

    # data_sampler["val_sampler"] 实际上包含的是【目标域（当前测试者）的查询集样本】
    # 这些样本作为"考题"，用于评估模型的分类性能
    val_iter = iter(data_sampler["val_sampler"])
    val_eeg_dset = data_dset["val_eeg_dset"]
    val_eye_dset = data_dset["val_eye_dset"]
    
    # 【PyTorch机制】使用torch.no_grad()上下文管理器禁用梯度计算。
    # 因为验证阶段不需要反向传播，关闭梯度计算可以大幅减少内存消耗并加速计算。
    with torch.no_grad():
        # 按照指定的迭代次数循环
        for it in range(parser.iterations):
            # ----- 获取支持集数据（来自目标域训练集） -----
            source_indices = next(train_iter)  # 获取当前batch的索引
            source_eeg_batch, source_labels = train_eeg_dset[source_indices]  # 加载EEG数据和标签
            source_eye_batch, _ = train_eye_dset[source_indices]  # 加载对应的眼动数据（忽略标签）

            # ----- 获取查询集数据（来自目标域验证集） -----
            val_indices = next(val_iter)
            val_eeg_batch, val_labels = val_eeg_dset[val_indices]
            val_eye_batch, _ = val_eye_dset[val_indices]

            # 【PyTorch机制】将所有数据从CPU内存移动到GPU显存，这样才能在GPU上进行并行计算
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
            # flag=1 是一个模式开关，指示protonet将第一个参数(source_fusion)作为支持集特征，
            # 第二个参数(val_fusion)作为查询集特征，计算每个查询样本到各类原型的距离
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


# =========================================================================
# 测试集测试阶段 (Test) - 逻辑与验证阶段基本相同，但使用测试集数据
# =========================================================================
def test(parser, data_dset, data_sampler, model, protonet):
    """
    在测试集上评估模型性能。
    
    测试阶段用于获得模型的最终性能指标。与验证阶段不同，测试集的数据在训练过程中
    完全未见过的，用于评估模型的真实泛化能力。
    
    参数:
        parser (argparse.Namespace): 命令行参数
        data_dset (dict): 数据集字典
        data_sampler (dict): 采样器字典
        model (nn.Module): 训练好的模型
        protonet (ProtoNet): 原型网络
    
    返回:
        tuple: (test_proto_loss, test_proto_acc, conf_matrix)
            - test_proto_loss (float): 测试集上的平均损失
            - test_proto_acc (float): 测试集上的平均准确率
            - conf_matrix (np.ndarray): 混淆矩阵，用于分析各类别的分类性能
    """
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    model.eval()  # 设置为评估模式

    test_proto_loss = 0.0
    test_proto_acc = 0.0

    # 【目标域支持集】仍然使用训练集的采样器（提供3个支持样本）
    train_iter = iter(data_sampler["train_sampler"])
    train_eeg_dset = data_dset["train_eeg_dset"]
    train_eye_dset = data_dset["train_eye_dset"]

    # 【目标域查询集】现在使用测试集的采样器（提供60个测试样本）
    test_target_iter = iter(data_sampler["test_sampler"])
    test_eeg_dset = data_dset["test_eeg_dset"]
    test_eye_dset = data_dset["test_eye_dset"]

    with torch.no_grad():  # 测试时关闭梯度计算
        for it in range(parser.iterations):
            # ----- 加载支持集数据（目标域训练集）-----
            source_indices = next(train_iter)
            source_eeg_batch, source_labels = train_eeg_dset[source_indices]
            source_eye_batch, _ = train_eye_dset[source_indices]

            # ----- 加载查询集数据（目标域测试集）-----
            test_indices = next(test_target_iter)
            test_eeg_batch, test_labels = test_eeg_dset[test_indices]
            test_eye_batch, _ = test_eye_dset[test_indices]

            # 数据迁移到GPU
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

            # ----- 计算损失、准确率和混淆矩阵 -----
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


# =========================================================================
# 主训练循环 (Train) - 整个模型的核心逻辑
# =========================================================================
def main(parser, data_dset, data_sampler, writer, early_stopping):
    """
    主训练函数，执行完整的训练、验证和测试流程。
    
    训练流程包含以下关键步骤：
    1. 初始化模型和优化器
    2. 对每个epoch：
       a. 使用源域数据（其他7个被试）训练模型，计算分类损失
       b. 计算GDD损失，对齐源域和目标域（当前被试）的特征分布
       c. 反向传播更新模型参数
       d. 在目标域的验证集上评估
       e. 检查早停条件
    3. 在目标域的测试集上评估最佳模型和最后一个模型
    
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
    device = torch.device("cuda:" + str(parser.cuda) if torch.cuda.is_available else "cpu")
    set_seed(parser.seed)

    # 1. 实例化主模型和原型网络分类器
    model = MyModel(eeg_input_dim, eye_input_dim, output_dim)
    protonet = ProtoNet()

    # 将模型移动到GPU（如果可用）
    model = model.to(device)
    protonet = protonet.to(device)

    # 2. 收集模型参数并初始化优化器
    params = list(model.parameters())
    optim = torch.optim.Adam(
        params=params,
        lr=parser.learning_rate
    )

    # 初始化最佳准确率记录
    best_acc = 0.0
    
    print("----------开始训练模型----------")
    
    # ==================== 训练主循环 ====================
    for epoch in range(1, parser.epochs + 1):
        print('========= Epoch: {} ========='.format(epoch))
        
        # 【PyTorch机制】设置为训练模式，启用BatchNorm和Dropout的训练行为
        model.train()
        protonet.train()

        # 初始化训练指标累加器
        train_loss = 0           # 总损失（分类损失 + GDD损失）
        train_gdd_loss = 0       # GDD域适应损失
        train_proto_loss_src = 0 # 源域分类损失
        train_proto_loss_tgt = 0 # 目标域分类损失（预留）
        train_proto_acc_src = 0  # 源域分类准确率
        train_proto_acc_tgt = 0  # 目标域分类准确率（预留）

        # 【角色匹配 - 极其重要】
        # train_source_iter: 源域数据（其他7个被试），包含63个样本（3个类别 × (1支持+20查询)）
        # train_iter: 目标域数据（当前被试），只包含3个支持样本，用于提取个体特征
        train_source_iter = iter(data_sampler["source_sampler"])
        train_iter = iter(data_sampler["train_sampler"])
        source_eeg_dset = data_dset["source_eeg_dset"]
        source_eye_dset = data_dset["source_eye_dset"]
        train_eeg_dset = data_dset["train_eeg_dset"]
        train_eye_dset = data_dset["train_eye_dset"]

        # 内层循环：控制每个epoch内的迭代次数
        for it in range(parser.iterations):
            # ----- 加载源域数据（教材数据）-----
            source_indices = next(train_source_iter)
            source_eeg_batch, source_labels = source_eeg_dset[source_indices]
            source_eye_batch, _ = source_eye_dset[source_indices]

            source_eeg_batch = source_eeg_batch.to(device)
            source_eye_batch = source_eye_batch.to(device)
            source_labels = source_labels.to(device)

            # ----- 加载目标域数据（风格参考数据）-----
            target_indices = next(train_iter)
            target_eeg_batch, target_labels = train_eeg_dset[target_indices]
            target_eye_batch, _ = train_eye_dset[target_indices]

            target_eeg_batch = target_eeg_batch.to(device)
            target_eye_batch = target_eye_batch.to(device)
            target_labels = target_labels.to(device)

            # ----- 前向传播：提取源域和目标域的多层特征 -----
            # model返回的是包含多层特征的列表（实际代码中是3层注意力特征）
            source_fusion = model(source_eeg_batch, source_eye_batch)
            target_fusion = model(target_eeg_batch, target_eye_batch)

            # ===== GDD损失计算（核心域适应部分） =====
            # 计算每一层特征的GDD损失，用于对齐源域和目标域的特征分布
            # zip的作用是让第1层特征对齐第1层，第2层对齐第2层，以此类推
            gdd_losses = [
                gdd(src, tgt)  # 计算两个特征分布之间的测地线距离
                for src, tgt in zip(source_fusion, target_fusion)
            ]  # 得到长度为3的列表 [gdd_loss1, gdd_loss2, gdd_loss3]

            # ----- 计算源域的分类损失（只有源域数据参与分类训练）-----
            # 使用源域特征的最后一层进行原型分类
            dist_src = protonet(
                source_fusion[-1],  # 使用最后一层特征
                source_labels,
                parser.classes_per_it_src,
                parser.num_support_src,
                parser.num_query_src
            )
            # 计算分类损失和准确率
            proto_loss_src, proto_acc_src, _ = prototypical_loss2(
                dist_src,
                parser.classes_per_it_src,
                parser.num_query_src,
                parser
            )

            # ===== GDD损失的动态加权 =====
            # gamma是一个随epoch动态变化的权重因子，控制GDD损失的重要性
            # 使用sigmoid-like曲线：前期gamma小（域适应权重低），后期gamma大（域适应权重高）
            # 这样设计是因为前期模型主要学习基本分类能力，后期才重点对齐个体差异
            gamma = 2 / (1 + np.exp(-10 * (epoch) / (parser.epochs))) - 1
            
            # 当前代码简化实现：只使用最后一层的GDD损失
            gdd_loss = gamma * gdd_losses[-1]
            
            # 最终的损失 = 分类损失 + 域适应损失
            loss = proto_loss_src + gdd_loss

            # 【PyTorch机制 - 网络反向传播标准三步曲】
            optim.zero_grad()  # 1. 清空上一轮计算的梯度（梯度会累加，不清空会导致错误）
            loss.backward()    # 2. 反向传播计算当前batch的梯度
            optim.step()       # 3. 根据梯度更新模型参数

            # 累加训练指标（用于打印和TensorBoard记录）
            train_loss += loss / parser.iterations
            train_gdd_loss += (gdd_loss) / parser.iterations
            train_proto_loss_src += proto_loss_src / parser.iterations
            train_proto_acc_src += proto_acc_src / parser.iterations

        # 记录训练指标到TensorBoard
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('gdd_loss', train_gdd_loss, epoch)
        writer.add_scalar('proto_loss_src', train_proto_loss_src, epoch)
        writer.add_scalar('proto_acc_src', train_proto_acc_src, epoch)

        # 打印训练指标
        print('Train Loss: {:.6f}\tmmd_loss:{:.6f}\tproto_loss_src:{:.6f}\tproto_loss_tgt:{:.6f}\tproto_acc_src:{:.6f}\tproto_acc_tgt:{:.6f}'.format(
            train_loss, train_gdd_loss, train_proto_loss_src, train_proto_loss_tgt, train_proto_acc_src, train_proto_acc_tgt))

        # ===== 验证阶段：评估当前epoch的模型在目标域验证集上的表现 =====
        val_proto_loss, val_proto_acc = val(parser, data_dset, data_sampler, model, protonet)
        
        # 更新最佳验证准确率
        if val_proto_acc > best_acc:
            best_acc = val_proto_acc

        # 【早停机制】如果验证准确率连续patience个epoch没有提升，则提前终止训练
        early_stopping(val_proto_acc, model)

        # 检查是否满足早停条件
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
    # 加载训练过程中验证准确率最高的模型参数进行测试
    model.load_state_dict(torch.load(early_stopping.best_model_path))
    best_test_loss, best_test_acc, best_conf_matrix = test(parser, data_dset, data_sampler, model, protonet)

    # 加载最后一个epoch的模型参数进行测试
    model.load_state_dict(torch.load(early_stopping.last_model_path))
    last_test_loss, last_test_acc, last_conf_matrix = test(parser, data_dset, data_sampler, model, protonet)

    # 选择表现更好的模型的结果
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


# =========================================================================
# 程序执行入口
# =========================================================================
if __name__ == "__main__":
    # 解析命令行参数
    parser = get_parser().parse_args()

    # ==================== 数据集配置 ====================
    # EEG数据文件名列表（每个session对应一组文件）
    # SEED-Franch数据集包含8个被试
    eeg_session_names = [
        ['1_1.npz', '2_1.npz', '3_1.npz', '4_1.npz', '5_1.npz', '6_1.npz', '7_1.npz', '8_1.npz'],  # Session 1
        ['1_2.npz', '2_2.npz', '3_2.npz', '4_2.npz', '5_2.npz', '6_2.npz', '7_2.npz', '8_2.npz'],  # Session 2
        ['1_3.npz', '2_3.npz', '3_3.npz', '4_3.npz', '5_3.npz', '6_3.npz', '7_3.npz', '8_3.npz']   # Session 3
    ]

    # 眼动数据文件夹名列表
    eye_session_names = [
        ['1_1', '2_1', '3_1', '4_1', '5_1', '6_1', '7_1', '8_1'],  # Session 1
        ['1_2', '2_2', '3_2', '4_2', '5_2', '6_2', '7_2', '8_2'],  # Session 2
        ['1_3', '2_3', '3_3', '4_3', '5_3', '6_3', '7_3', '8_3']   # Session 3
    ]

    # 数据路径配置（需要根据实际环境修改）
    eeg_path = parser.eeg_path
    eye_path = parser.eye_path

    # ==================== 跨session实验 ====================
    session_acc = []  # 记录每个session的平均准确率
    total_conf_matrix = np.zeros((3, 3))  # 累积所有被试的混淆矩阵
    
    # 大循环：遍历3个不同的实验周期（Session）
    for session in range(0, 3):
        sum_acc = 0.0
        mean_acc = 0.0
        acc_list = []  # 记录当前session中每个被试的准确率
        
        # 【PyTorch机制】手动清空GPU缓存，避免显存碎片累积导致显存不足
        torch.cuda.empty_cache()
        idx = 0
        print('%%%%%%%%%% Session: {} %%%%%%%%%%'.format(session))
        
        # ==================== LOSO (留一被试交叉验证) ====================
        # 中循环：总共有8个被试，每次留一个作为目标域，其余7个作为源域
        for subject in range(0, 8):
            idx += 1
            # 源域：除当前被试外的所有其他被试（7个）
            src_idx = [i for i in range(8) if i != subject]
            # 目标域：当前被试
            tar_idx = [subject]
            
            print(f"源域被试索引: {src_idx}")
            print(f"目标域被试索引: {tar_idx}")
            print('%%%%%%%%%% 目标被试: {} %%%%%%%%%%'.format(subject))

            # ----- 加载数据 -----
            # 源域数据（其他7个被试，用于训练），使用"full"模式加载全部数据
            source_eeg_sample, source_eye_sample, source_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, src_idx, "full"
            )
            
            # 目标域数据（当前被试）划分为训练/验证/测试集
            # 训练集：3个支持样本（每个类别1个）
            train_eeg_sample, train_eye_sample, train_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, tar_idx, "train"
            )
            # 验证集：60个验证样本
            val_eeg_sample, val_eye_sample, val_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, tar_idx, "val"
            )
            # 测试集：60个测试样本
            test_eeg_sample, test_eye_sample, test_label = load4data(
                parser, eeg_path, eye_path, eeg_session_names, eye_session_names,
                session, tar_idx, "test"
            )

            # 【PyTorch机制】创建TensorDataset，将特征张量和标签张量配对封装成数据集
            source_eeg_dset = torch.utils.data.TensorDataset(source_eeg_sample, source_label)
            source_eye_dset = torch.utils.data.TensorDataset(source_eye_sample, source_label)

            train_eeg_dset = torch.utils.data.TensorDataset(train_eeg_sample, train_label)
            train_eye_dset = torch.utils.data.TensorDataset(train_eye_sample, train_label)

            val_eeg_dset = torch.utils.data.TensorDataset(val_eeg_sample, val_label)
            val_eye_dset = torch.utils.data.TensorDataset(val_eye_sample, val_label)

            test_eeg_dset = torch.utils.data.TensorDataset(test_eeg_sample, test_label)
            test_eye_dset = torch.utils.data.TensorDataset(test_eye_sample, test_label)

            # 打印数据集大小信息（用于调试）
            print(f"源域EEG数据集大小: {len(source_eeg_dset)}")
            print(f"目标域训练集EEG大小: {len(train_eeg_dset)}")
            print(f"目标域验证集EEG大小: {len(val_eeg_dset)}")
            print(f"目标域测试集EEG大小: {len(test_eeg_dset)}")

            # 【核心策略：实例化小样本学习专用采样器】
            # PrototypicalBatchSampler确保每次取出的batch符合"N-way K-shot"格式
            # 参数：标签、类别数(3)、每个类别的样本数、迭代次数
            source_sampler = PrototypicalBatchSampler(
                source_label, 3,
                parser.num_support_src + parser.num_query_src,  # 每个类别的总样本数（1支持+20查询）
                parser.iterations
            )
            train_sampler = PrototypicalBatchSampler(
                train_label, 3,
                parser.num_support_tgt,  # 每个类别1个支持样本
                parser.iterations
            )
            val_sampler = PrototypicalBatchSampler(
                val_label, 3,
                parser.num_query_tgt,  # 每个类别20个查询样本
                parser.iterations
            )
            test_sampler = PrototypicalBatchSampler(
                test_label, 3,
                parser.num_query_tgt,  # 每个类别20个查询样本
                parser.iterations
            )

            # 整理数据集和采样器到字典中，方便传递给训练函数
            data_set = {
                "source_eeg_dset": source_eeg_dset,
                "source_eye_dset": source_eye_dset,
                "train_eeg_dset": train_eeg_dset,
                "train_eye_dset": train_eye_dset,
                "val_eeg_dset": val_eeg_dset,
                "val_eye_dset": val_eye_dset,
                "test_eeg_dset": test_eeg_dset,
                "test_eye_dset": test_eye_dset
            }

            data_sampler = {
                "source_sampler": source_sampler,
                "train_sampler": train_sampler,
                "val_sampler": val_sampler,
                "test_sampler": test_sampler
            }

            # 初始化TensorBoard写入器，用于记录训练过程
            writer = SummaryWriter("data/tensorboard/experiment/session" + str(session) + "CAMP/" + "target" + str(subject))
            
            # 创建早停实例
            early_stopping = EarlyStoppingAccuracy(
                patience=parser.patience,
                verbose=True,
                individual_id=str(subject),
                session_id=str(session)
            )
            
            # >>> 开始训练当前被试的模型 <<<
            best_test_loss, best_test_acc, last_test_loss, last_test_acc, conf_matrix = main(
                parser, data_set, data_sampler, writer, early_stopping
            )
            
            # 累积混淆矩阵
            total_conf_matrix += conf_matrix
            
            # 打印当前被试的结果
            print(f"\n被试 {subject}: 最佳测试准确率 = {best_test_acc:.4f}, 最后测试准确率 = {last_test_acc:.4f}")
            
            # 取最佳和最后模型中准确率更高的一个
            sum_acc += max(best_test_acc, last_test_acc)
            mean_acc = sum_acc / idx
            acc_list.append(max(best_test_acc, last_test_acc).item())
            
            print(f"\n被试 {subject}: 当前平均准确率 = {mean_acc:.4f}")
            print(f"被试 {subject}: 准确率列表 = {acc_list}\n")
            
            # 关闭TensorBoard写入器
            writer.close()
            
        # 打印当前session的统计结果
        print(f"Session {session}: 平均准确率 = {sum_acc / idx:.4f}")
        print(f"准确率列表: {acc_list}\n")
        session_acc.append((sum_acc / idx).item())
        print(f"各Session准确率: {session_acc}")
    
    # 打印最终结果（所有session的平均）
    print(f"各Session准确率: {session_acc}")
    print(f"最终平均准确率: {(session_acc[0] + session_acc[1] + session_acc[2]) / 3:.4f}")