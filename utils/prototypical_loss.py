# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from sklearn.metrics import confusion_matrix

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x(queries): N x D
    # y(prototypes): M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support, opt):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    device = "cuda:" + str(opt.cuda) if torch.cuda.is_available() else "cpu"

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])  # compute prototypes of all class
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input[query_idxs]


    dists = euclidean_dist(query_samples, prototypes)  # compute distance between each query and each prototype

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)\


    target_inds = torch.arange(0, n_classes).to(device)
    target_inds = target_inds.view(n_classes, 1, 1).to(device)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    # print('y_hat:', y_hat.cpu().detach().numpy())
    # print('target_inds:', target_inds.squeeze().cpu().detach().numpy())
    # print('correct count:', y_hat.eq(target_inds.squeeze()).float().sum().cpu().detach().numpy())

    return loss_val,  acc_val


def prototypical_loss2(dists, n_classes, n_query, opt):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    device = "cuda:" + str(opt.cuda) if torch.cuda.is_available() else "cpu"

    #直接得到对数概率
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    #由于每次迭代时，我们知道查询样本的类别是如何组织的（因为是通过原型网络进行分类，每个类的查询样本是有组织的）
    #因此可以通过这些类别索引来确定真实类别标签
    target_inds = torch.arange(0, n_classes).to(device)
    target_inds = target_inds.view(n_classes, 1, 1).to(device)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)

    #gather(dim=2, index)：从 log_p_y 的第 2 维（即类别维度）中根据 target_inds 提供的索引来选择相应的值
    #squeeze() 和 view(-1)：移除不必要的维度并展平张量，方便计算平均损失。
    #mean()：对所有查询样本的负对数概率取平均，计算整体的交叉熵损失。
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    """
    首先，通过 log_p_y.max(2) 获得每个查询样本的预测类别 y_hat。
    然后，使用 y_hat.eq(target_inds.squeeze()) 将预测类别与真实类别进行逐元素比较，生成一个布尔张量。
    通过 .float() 将布尔张量转化为浮点数，True 转换为 1.0，False 转换为 0.0。
    最后，通过 .mean() 计算预测正确的样本比例，即为准确率。
    """
    _, y_hat = log_p_y.max(2)

    # 将预测结果和真实标签展平，方便计算混淆矩阵
    y_hat_flat = y_hat.view(-1).cpu().numpy()  # 转为 numpy 数组
    target_flat = target_inds.squeeze().reshape(-1).cpu().numpy()  # 转为 numpy 数组

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(target_flat, y_hat_flat)

    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val, conf_matrix

def prototypical_loss_aggregate(dists_list, n_classes, n_query, opt):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    device = "cuda:" + str(opt.cuda) if torch.cuda.is_available() else "cpu"

    # aggregate all dists computed by target querys and each source supports
    weighted_dist_list = []
    for i in range(opt.num_distill_source):
        src_weight = 1 / (i + 1)
        weighted_dist_list.append(src_weight * dists_list[i])
    weighted_dists = torch.stack(weighted_dist_list).view(opt.num_distill_source, n_classes*n_query, n_classes)  # (num_source, n_classes*n_query, n_classes)
    agg_dists = torch.sum(weighted_dists, dim=0)  # (n_classes*n_query, n_classes)

    log_p_y = F.log_softmax(-agg_dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).to(device)
    target_inds = target_inds.view(n_classes, 1, 1).to(device)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val
