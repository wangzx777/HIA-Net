# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from sklearn.metrics import confusion_matrix


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    原型损失类 - 对应论文2.D节的Few-Shot分类损失
    
    功能：封装原型损失函数，使其可以像标准的PyTorch损失函数一样使用
    '''
    def __init__(self, n_support):
        """
        Args:
            n_support: support样本数K（每个类的support样本数量）
                      论文中K=1,5,10,20对应不同的shot设置
        """
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        """
        Args:
            input: 模型输出的特征，形状 (total_samples, feature_dim)
            target: 对应的标签，形状 (total_samples,)
        
        Returns:
            loss, acc: 损失值和准确率
        """
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    计算欧氏距离 - 对应论文公式(7)中的dist函数
    
    Args:
        x: 查询样本特征，形状 (n_query * n_classes, feature_dim)
        y: 原型特征，形状 (n_classes, feature_dim)
    
    Returns:
        距离矩阵，形状 (n_query * n_classes, n_classes)
    '''
    # x(queries): N x D
    # y(prototypes): M x D
    n = x.size(0)  # 查询样本总数 = n_classes * n_query
    m = y.size(0)  # 类别数 = n_classes
    d = x.size(1)  # 特征维度
    if d != y.size(1):
        raise Exception

    # 扩展维度以便计算所有配对距离
    x = x.unsqueeze(1).expand(n, m, d)  # (n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)  # (n, m, d)

    # 计算欧氏距离的平方
    return torch.pow(x - y, 2).sum(2)  # (n, m)


def prototypical_loss(input, target, n_support, opt):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    
    原型损失函数 - 对应论文公式(7)的分类损失
    
    Args:
    - input: 模型输出的特征，形状 (total_samples, feature_dim)
    - target: 对应的标签，形状 (total_samples,)
    - n_support: support样本数K
    - opt: 配置参数，包含cuda设备信息
    
    Returns:
    - loss_val: 交叉熵损失值
    - acc_val: 分类准确率
    '''
    device = "cuda:" + str(opt.cuda) if torch.cuda.is_available() else "cpu"

    def supp_idxs(c):
        """
        为每个类c提取前n_support个样本的索引
        
        示例：target = [0,0,0,1,1,1,2,2,2], n_support=1
        对c=0: target.eq(0) -> [True,True,True,False,False,False,False,False,False]
               .nonzero() -> [[0],[1],[2]]
               [:1] -> [[0]]
               .squeeze(1) -> [0]
        返回索引0
        """
        # FIXME when torch will support where as np
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target)  # 获取所有类别 [0,1,2]
    n_classes = len(classes)  # 类别数N（论文中N=3）
    
    # FIXME when torch will support where as np
    # 计算每个类的query样本数
    n_query = target.eq(classes[0].item()).sum().item() - n_support  # 每个类的总样本数 - support数

    # ============ 步骤1：提取support样本并计算原型 ============
    support_idxs = list(map(supp_idxs, classes))  # 每个类的support索引列表
    
    # 计算原型：对每个类，对support样本的特征取平均
    # input[idx_list] 提取每个类的support样本特征
    # .mean(0) 在样本维度上取平均，得到原型向量
    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])  # (n_classes, feature_dim)

    # ============ 步骤2：提取query样本 ============
    # 为每个类提取n_support之后的样本作为query
    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input[query_idxs]  # (n_classes * n_query, feature_dim)

    # ============ 步骤3：计算距离和损失 ============
    # 计算每个query样本到每个原型的距离
    dists = euclidean_dist(query_samples, prototypes)  # (n_classes*n_query, n_classes)

    # 将距离转换为概率：p = exp(-dist) / sum(exp(-dist))
    # 对应论文公式(7)：p(y=i|x) = exp(-dist) / sum(exp(-dist))
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)  # (n_classes, n_query, n_classes)

    # ============ 步骤4：构建真实标签 ============
    # 由于query样本是按类别顺序组织的，真实标签就是[0,1,2]重复n_query次
    target_inds = torch.arange(0, n_classes).to(device)  # [0,1,2]
    target_inds = target_inds.view(n_classes, 1, 1).to(device)  # (3,1,1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)  # (3,20,1)

    # ============ 步骤5：计算损失和准确率 ============
    # 交叉熵损失：-log(p(y_true|x))
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    
    # 预测类别：选择概率最大的类
    _, y_hat = log_p_y.max(2)  # (n_classes, n_query)
    
    # 计算准确率
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val


def prototypical_loss2(dists, n_classes, n_query, opt):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    
    原型损失函数V2 - 直接接收距离矩阵作为输入
    
    Args:
    - dists: 距离矩阵，形状 (n_classes * n_query, n_classes)
    - n_classes: 类别数N
    - n_query: 每个类的query样本数
    - opt: 配置参数
    
    Returns:
    - loss_val: 损失值
    - acc_val: 准确率
    - conf_matrix: 混淆矩阵
    '''
    device = "cuda:" + str(opt.cuda) if torch.cuda.is_available() else "cpu"

    # 直接得到对数概率
    # 对应论文公式(7)：p = exp(-dist) / sum(exp(-dist))
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    # 由于每次迭代时，我们知道查询样本的类别是如何组织的（因为是通过原型网络进行分类，每个类的查询样本是有组织的）
    # 因此可以通过这些类别索引来确定真实类别标签
    target_inds = torch.arange(0, n_classes).to(device)  # [0,1,2]
    target_inds = target_inds.view(n_classes, 1, 1).to(device)  # (3,1,1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)  # (3,20,1)

    # gather(dim=2, index)：从 log_p_y 的第 2 维（即类别维度）中根据 target_inds 提供的索引来选择相应的值
    # squeeze() 和 view(-1)：移除不必要的维度并展平张量，方便计算平均损失。
    # mean()：对所有查询样本的负对数概率取平均，计算整体的交叉熵损失。
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    """
    首先，通过 log_p_y.max(2) 获得每个查询样本的预测类别 y_hat。
    然后，使用 y_hat.eq(target_inds.squeeze()) 将预测类别与真实类别进行逐元素比较，生成一个布尔张量。
    通过 .float() 将布尔张量转化为浮点数，True 转换为 1.0，False 转换为 0.0。
    最后，通过 .mean() 计算预测正确的样本比例，即为准确率。
    """
    _, y_hat = log_p_y.max(2)  # (n_classes, n_query)

    # 将预测结果和真实标签展平，方便计算混淆矩阵
    y_hat_flat = y_hat.view(-1).cpu().numpy()  # 转为 numpy 数组
    target_flat = target_inds.squeeze().reshape(-1).cpu().numpy()  # 转为 numpy 数组

    # 计算混淆矩阵 - 对应论文图2的混淆矩阵可视化
    conf_matrix = confusion_matrix(target_flat, y_hat_flat)

    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val, conf_matrix


def prototypical_loss_aggregate(dists_list, n_classes, n_query, opt):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    
    聚合原型损失函数 - 用于多源域蒸馏
    
    Args:
    - dists_list: 距离矩阵列表，每个元素形状 (n_classes * n_query, n_classes)
    - n_classes: 类别数N
    - n_query: 每个类的query样本数
    - opt: 配置参数，包含num_distill_source（源域蒸馏数量）
    
    Returns:
    - loss_val: 损失值
    - acc_val: 准确率
    '''
    device = "cuda:" + str(opt.cuda) if torch.cuda.is_available() else "cpu"

    # aggregate all dists computed by target querys and each source supports
    # 聚合所有源域计算出的距离，每个源域的权重与其序号成反比（1/(i+1)）
    weighted_dist_list = []
    for i in range(opt.num_distill_source):
        src_weight = 1 / (i + 1)  # 权重递减：1, 1/2, 1/3, ...
        weighted_dist_list.append(src_weight * dists_list[i])
    
    # 堆叠并求和得到聚合距离
    weighted_dists = torch.stack(weighted_dist_list).view(opt.num_distill_source, n_classes * n_query, n_classes)  # (num_source, n_classes*n_query, n_classes)
    agg_dists = torch.sum(weighted_dists, dim=0)  # (n_classes*n_query, n_classes)

    # 后续计算与prototypical_loss2相同
    log_p_y = F.log_softmax(-agg_dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).to(device)
    target_inds = target_inds.view(n_classes, 1, 1).to(device)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val


"""
=============================================================================
数据流形状变化追踪
=============================================================================

prototypical_loss 函数：
--------------------------------------------------------------------------------
输入:
    input: (total_samples, feature_dim)  # total_samples = n_classes * (n_support + n_query)
    target: (total_samples,)
    
步骤1: 提取support索引
    support_idxs: 列表，每个元素形状 (n_support,)
    
步骤2: 计算原型
    prototypes: (n_classes, feature_dim)  # 对每个类的support样本取平均
    
步骤3: 提取query样本
    query_samples: (n_classes * n_query, feature_dim)
    
步骤4: 计算距离
    dists: (n_classes * n_query, n_classes)
    
步骤5: softmax转换为概率
    log_p_y: (n_classes, n_query, n_classes)
    
步骤6: 构建真实标签
    target_inds: (n_classes, n_query, 1)
    
步骤7: 计算损失和准确率
    loss_val: 标量
    acc_val: 标量

典型数值示例 (3-way 5-shot):
--------------------------------------------------------------------------------
- n_classes = 3, n_support = 5, n_query = 20
- total_samples = 3 * (5 + 20) = 75
- input: (75, 256)
- prototypes: (3, 256)
- query_samples: (60, 256)
- dists: (60, 3)
- log_p_y: (3, 20, 3)
- target_inds: (3, 20, 1)
- loss_val: 标量 (例如 0.85)
- acc_val: 标量 (例如 0.72)

=============================================================================
代码与论文对应关系
=============================================================================

| 代码组件 | 论文对应部分 | 功能说明 |
|---------|------------|---------|
| `prototypical_loss` | 2.D节 公式 L_CE | 原型网络分类损失 |
| `euclidean_dist` | 公式(7)中的dist | 计算欧氏距离 |
| `prototypes.mean(0)` | 公式 p_i = 1/|S_i| * sum(f(x)) | 计算类别原型 |
| `F.log_softmax(-dists)` | 公式(7) p(y=i|x) | 距离转换为概率 |
| `confusion_matrix` | 图2 混淆矩阵 | 可视化分类结果 |
| `prototypical_loss_aggregate` | 可能的多源域扩展 | 聚合多个源域的知识 |

=============================================================================
关键点解释
=============================================================================

1. **为什么用负距离计算概率？**
   - 公式(7): p(y=i|x) = exp(-dist) / sum(exp(-dist))
   - 距离越小（样本越接近原型），概率越大
   - 符合直觉：离原型越近，属于该类别的可能性越大

2. **为什么按类别组织query样本？**
   - target_inds = [0,1,2]重复n_query次
   - 因为query样本是按类别顺序提取的
   - 前n_query个属于类0，接着n_query个属于类1，...

3. **gather函数的作用？**
   - log_p_y: (3,20,3) 每个样本在3个类别上的log概率
   - target_inds: (3,20,1) 真实类别索引
   - gather(2, target_inds) 提取每个样本真实类别对应的log概率
   - 用于计算交叉熵损失

4. **prototypical_loss2与prototypical_loss的区别？**
   - loss: 内部计算距离矩阵
   - loss2: 直接接收距离矩阵作为输入
   - loss2更灵活，可以用于已经计算好距离的场景

5. **prototypical_loss_aggregate的作用？**
   - 可能用于多源域学习
   - 对多个源域计算的距离加权聚合
   - 权重递减：越靠前的源域权重越大
"""