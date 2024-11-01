import torch.nn as nn
from torch.nn import functional as F
import torch


class ProtoNet(nn.Module):
    '''
    Prototypes of each class in support data
    '''

    def __init__(self):
        super(ProtoNet, self).__init__()

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x(queries): N x D (n_classes*n_query, dim)
        # y(prototypes): M x D (n_classes, dim)
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)  # (n_classes*n_query, n_classes, dim)
        y = y.unsqueeze(0).expand(n, m, d)  # (n_classes*n_query, n_classes, dim)

        return torch.pow(x - y, 2).sum(2)  # (n_classes*n_query, n_classes)

    def forward(self, x, y, n_classes, n_support=None, n_query=None, flag=0):
        if flag == 0:  # support and query all come from source/target
            def supp_idxs(c):
                """
                为每个类 c 提取前 n_support 个样本的索引

                假设 y = torch.tensor([0, 1, 2, 1, 2, 0, 1]) 且 n_support = 2，函数的作用如下：
                如果调用 supp_idxs(1)，那么：
                y.eq(1) 返回 [False, True, False, True, False, False, True]
                .nonzero() 返回 [1, 3, 6]
                [:n_support] 截取前 2 个，结果为 [1, 3]
                .squeeze(1) 后，得到 [1, 3]，表示标签为 1 的前两个样本的索引。
                """
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            classes = torch.unique(y)
            # 为每个类 c 提取前 n_support 个样本的索引
            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
            query_samples = x[query_idxs]  # (n_classes *n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

        else:  # support and query come from source and target respectively
            # support来自source,query来自target
            data_src, data_tgt = x[0], x[1]
            label_src, label_tgt = y[0], y[1]
            classes = torch.unique(label_src)
            # 为每个类 c 提取前 n_support 个样本的索引
            support_idxs = list(map(lambda c: label_src.eq(c).nonzero()[:n_support], classes))
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, n_support, -1)  # (n_classes, n_support, dim)
            # 为每个类 c 提取前 n_query 个样本的索引
            query_idxs = list(map(lambda c: label_tgt.eq(c).nonzero()[:n_query], classes))
            query_idxs = torch.stack(query_idxs).view(-1)
            query_samples = data_tgt[query_idxs]  # (n_classes*n_query, dim)

        # 对每个类别取平均值作为原型
        support_prototypes = support_samples.mean(1)  # (n_classes, dim)
        # 返回每个query到原型的欧氏距离
        dists = self.euclidean_dist(query_samples, support_prototypes)
        return dists


class ProtoHybridAttention(nn.Module):
    '''
    Prototypes of each class in support data with instance and feature attention mechanism
    Reference paper: <Hybrid Attention-Based Prototypical Networks for Noisy
    Few-Shot Relation Classification>
    '''

    def __init__(self, n_support, h_dim):
        super(ProtoHybridAttention, self).__init__()
        self.h_dim = h_dim
        self.drop = nn.Dropout()
        # for instance-level attention
        self.fc = nn.Linear(h_dim, h_dim)

        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (n_support, 1), padding=(n_support // 2, 0))
        self.init_weights(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, (n_support, 1), padding=(n_support // 2, 0))
        self.init_weights(self.conv2)
        self.conv_final = nn.Conv2d(64, 1, (n_support, 1), stride=(n_support, 1))
        self.init_weights(self.conv_final)

    def init_weights(self, layer):
        # nn.init.xavier_uniform(layer.weight)
        nn.init.xavier_normal(layer.weight)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(1), 2, score)

    def forward(self, x, y, n_classes, n_support, n_query):
        # support samples: (n_classes*n_support, dim)
        # query samples: (n_classes *nery, dim)
        classes = torch.unique(y)

        def supp_idxs(c):
            # FIXME when torch will support where as np
            return y.eq(c).nonzero()[:n_support].squeeze(1)

        support_idxs = list(map(supp_idxs, classes))
        query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
        query_samples = x[query_idxs]  # (n_classes *n_query, dim)
        support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
        support = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

        # feature-level attention
        fea_att_score = support.view(n_classes, 1, n_support, self.h_dim)  # (n_classes, 1, n_support, dim)
        fea_att_score = F.tanh(self.conv1(fea_att_score))  # (n_classes, 32, n_support, dim)
        fea_att_score = F.tanh(self.conv2(fea_att_score))  # (n_classes, 64, n_support, dim)

        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score)  # (n_classes, 1, 1, dim)

        fea_att_score = F.tanh(fea_att_score)
        fea_att_score = fea_att_score.view(n_classes, self.h_dim).unsqueeze(0)  # (1, n_classes, dim)

        # instance-level attention
        support_ = support.unsqueeze(0).expand(n_classes * n_query, -1, -1,
                                               -1)  # (n_classes*n_query, n_classes, n_support, dim)
        support_for_att = self.fc(support_)

        query_ = query_samples.unsqueeze(1).unsqueeze(2).expand(-1, n_classes, n_support,
                                                                -1)  # (n_classes*n_query, n_classes, n_support, dim)
        query_for_att = self.fc(query_)

        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1),
                                  dim=-1)  # (n_classes*n_query, n_classes, n_support)
        support_proto = (support * ins_att_score.unsqueeze(3).expand(-1, -1, -1, self.h_dim)).sum(
            2)  # (n_classes*n_query, n_classes, dim)

        dists = self.__batch_dist__(support_proto, query_samples, fea_att_score)  # (n_classes*n_query, n_classes)

        return dists


class ProtoInstanceAttention(nn.Module):
    '''
    Prototypes of each class in support data with instance attention mechanism
    '''

    def __init__(self, n_support, h_dim):
        super(ProtoInstanceAttention, self).__init__()
        self.h_dim = h_dim
        self.drop = nn.Dropout()
        # for instance-level attention
        self.fc = nn.Linear(h_dim, h_dim)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(1), 2, score)

    def forward(self, x, y, n_classes, n_support, n_query, flag=0):
        # support samples: (n_classes*n_support, dim)
        # query samples: (n_classes *n_query, dim)
        if flag == 0:  # support and query all come from source/target
            def supp_idxs(c):
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            classes = torch.unique(y)
            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
            query_samples = x[query_idxs]  # (n_classes *n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)
        else:  # support and query come from source and target respectively
            data_src, data_tgt = x[0], x[1]
            label_src, label_tgt = y[0], y[1]
            classes = torch.unique(label_src)
            support_idxs = list(map(lambda c: label_src.eq(c).nonzero()[:n_support], classes))
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

            query_idxs = list(map(lambda c: label_tgt.eq(c).nonzero()[:n_query], classes))
            query_idxs = torch.stack(query_idxs).view(-1)
            query_samples = data_tgt[query_idxs]  # (n_classes*n_query, dim)

        # instance-level attention
        support_ = support_samples.unsqueeze(0).expand(n_classes * n_query, -1, -1,
                                                       -1)  # (n_classes*n_query, n_classes, n_support, dim)
        support_for_att = self.fc(support_)

        query_ = query_samples.unsqueeze(1).unsqueeze(2).expand(-1, n_classes, n_support,
                                                                -1)  # (n_classes*n_query, n_classes, n_support, dim)
        query_for_att = self.fc(query_)

        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1),
                                  dim=-1)  # (n_classes*n_query, n_classes, n_support)
        support_proto = (support_samples * ins_att_score.unsqueeze(3).expand(-1, -1, -1, self.h_dim)).sum(
            2)  # (n_classes*n_query, n_classes, dim)

        dists = self.__batch_dist__(support_proto, query_samples)  # (n_classes*n_query, n_classes)
        return dists


class ProtoInstanceMultiCrossAttention(nn.Module):
    '''
    Prototypes of each class in support data with instance attention mechanism
    '''

    def __init__(self, n_support, h_dim):
        super(ProtoInstanceMultiCrossAttention, self).__init__()
        self.h_dim = h_dim
        self.drop = nn.Dropout()
        # for instance-level multi cross attention
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(4 * h_dim, h_dim)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(1), 2, score)

    def forward(self, x, y, n_classes, n_support=None, n_query=None, flag=0):
        # support samples: (n_classes*n_support, dim)
        # query samples: (n_classes *n_query, dim)
        classes = torch.unique(y)
        num_support = n_support
        num_query = n_query
        if flag == 0:  # support and query all come from source/target
            def supp_idxs(c):
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[num_support:], classes))).view(-1)
            query_samples = x[query_idxs]  # (n_classes *n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, num_support, -1)  # (n_classes, n_support, dim)
        else:  # support and query come from source and target respectively
            data_src, query_samples = x[0], x[1]
            num_query = int(len(query_samples) / n_classes)
            support_idxs = list(map(lambda c: y.eq(c).nonzero(), classes))
            num_support = len(support_idxs[0])
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, num_support, -1)

        # instance-level multi cross attention
        support_ = support_samples.unsqueeze(0).expand(n_classes * num_query, -1, -1,
                                                       -1)  # (n_classes*n_query, n_classes, n_support, dim)
        support_for_att = self.fc1(support_)

        query_ = query_samples.unsqueeze(1).unsqueeze(2).expand(-1, n_classes, num_support,
                                                                -1)  # (n_classes*n_query, n_classes, n_support, dim)
        query_for_att = self.fc1(query_)

        diff_support_query_1 = torch.abs(
            support_for_att - query_for_att)  # # (n_classes*n_query, n_classes, n_support, dim)
        diff_support_query_2 = support_for_att * query_for_att  # (n_classes*n_query, n_classes, n_support, dim)

        mca = torch.cat((support_for_att, query_for_att, diff_support_query_1, diff_support_query_2),
                        dim=-1)  # (n_classes*n_query, n_classes, n_support, 4*dim)
        mca_ = self.fc2(mca)  # (n_classes*n_query, n_classes, n_support, dim)

        ins_mca_score = F.softmax(torch.tanh(mca_).sum(-1), dim=-1)  # (n_classes*n_query, n_classes, n_support)
        support_proto = (support_samples * ins_mca_score.unsqueeze(3).expand(-1, -1, -1, self.h_dim)).sum(
            2)  # (n_classes*n_query, n_classes, dim)

        dists = self.__batch_dist__(support_proto, query_samples)  # (n_classes*n_query, n_classes)
        return dists


class ProtoInstanceMutualAttention(nn.Module):
    '''
    Prototypes of each class in support data with instance attention mechanism
    '''

    def __init__(self, n_support, h_dim):
        super(ProtoInstanceMutualAttention, self).__init__()
        self.h_dim = h_dim
        self.drop = nn.Dropout()
        # for instance-level multi cross attention
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(2 * h_dim, h_dim)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(1), 2, score)

    def forward(self, x, y, n_classes, n_support=None, n_query=None, flag=0):
        # support samples: (n_classes*n_support, dim)
        # query samples: (n_classes *n_query, dim)
        classes = torch.unique(y)
        num_support = n_support
        num_query = n_query
        if flag == 0:  # support and query all come from source/target
            def supp_idxs(c):
                # FIXME when torch will support where as np
                return y.eq(c).nonzero()[:n_support].squeeze(1)

            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[num_support:], classes))).view(-1)
            query_samples = x[query_idxs]  # (n_classes *n_query, dim)
            support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
            support_samples = support_samples.view(n_classes, num_support, -1)  # (n_classes, n_support, dim)
        else:  # support and query come from source and target respectively
            data_src, query_samples = x[0], x[1]
            num_query = int(len(query_samples) / n_classes)
            support_idxs = list(map(lambda c: y.eq(c).nonzero(), classes))
            num_support = len(support_idxs[0])
            support_idxs = torch.stack(support_idxs).view(-1)
            support_samples = data_src[support_idxs].view(n_classes, num_support, -1)

        # instance-level multi cross attention
        support_ = support_samples.unsqueeze(0).expand(n_classes * num_query, -1, -1,
                                                       -1)  # (n_classes*n_query, n_classes, n_support, dim)
        support_for_att = self.fc1(support_)

        query_ = query_samples.unsqueeze(1).unsqueeze(2).expand(-1, n_classes, num_support,
                                                                -1)  # (n_classes*n_query, n_classes, n_support, dim)
        query_for_att = self.fc1(query_)

        diff_support_query_1 = torch.abs(
            support_for_att - query_for_att)  # # (n_classes*n_query, n_classes, n_support, dim)
        diff_support_query_2 = support_for_att * query_for_att  # (n_classes*n_query, n_classes, n_support, dim)

        mca = torch.cat((diff_support_query_1, diff_support_query_2),
                        dim=-1)  # (n_classes*n_query, n_classes, n_support, 2*dim)
        mca_ = self.fc2(mca)  # (n_classes*n_query, n_classes, n_support, dim)

        ins_mca_score = F.softmax(torch.tanh(mca_).sum(-1), dim=-1)  # (n_classes*n_query, n_classes, n_support)
        support_proto = (support_samples * ins_mca_score.unsqueeze(3).expand(-1, -1, -1, self.h_dim)).sum(
            2)  # (n_classes*n_query, n_classes, dim)

        dists = self.__batch_dist__(support_proto, query_samples)  # (n_classes*n_query, n_classes)
        return dists


class ProtoFeatureAttention(nn.Module):
    '''
    Prototypes of each class in support data with feature attention mechanism
    Reference paper: <Hybrid Attention-Based Prototypical Networks for Noisy
    Few-Shot Relation Classification>
    '''

    def __init__(self, n_support, h_dim):
        super(ProtoFeatureAttention, self).__init__()
        self.h_dim = h_dim
        self.drop = nn.Dropout()

        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (n_support, 1), padding=(n_support // 2, 0))
        self.init_weights(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, (n_support, 1), padding=(n_support // 2, 0))
        self.init_weights(self.conv2)
        self.conv_final = nn.Conv2d(64, 1, (n_support, 1), stride=(n_support, 1))
        self.init_weights(self.conv_final)

    def init_weights(self, layer):
        # nn.init.xavier_uniform(layer.weight)
        nn.init.xavier_normal(layer.weight)

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(1), 2, score)

    def forward(self, x, y, n_classes, n_support, n_query):
        # support samples: (n_classes*n_support, dim)
        # query samples: (n_classes *nery, dim)
        classes = torch.unique(y)

        def supp_idxs(c):
            # FIXME when torch will support where as np
            return y.eq(c).nonzero()[:n_support].squeeze(1)

        support_idxs = list(map(supp_idxs, classes))
        query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
        query_samples = x[query_idxs]  # (n_classes *n_query, dim)
        support_samples = x[torch.stack(support_idxs).view(-1)]  # (n_classes*n_support, dim)
        support = support_samples.view(n_classes, n_support, -1)  # (n_classes, n_support, dim)

        # feature-level attention
        fea_att_score = support.view(n_classes, 1, n_support, self.h_dim)  # (n_classes, 1, n_support, dim)
        fea_att_score = F.tanh(self.conv1(fea_att_score))  # (n_classes, 32, n_support, dim)
        fea_att_score = F.tanh(self.conv2(fea_att_score))  # (n_classes, 64, n_support, dim)

        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score)  # (n_classes, 1, 1, dim)

        fea_att_score = F.tanh(fea_att_score)
        fea_att_score = fea_att_score.view(n_classes, self.h_dim).unsqueeze(0)  # (1, n_classes, dim)

        support_proto = torch.stack(
            [x[idx_list].mean(0) for idx_list in support_idxs])  # compute prototypes of all class
        dists = self.__batch_dist__(support_proto, query_samples, fea_att_score)  # (n_classes*n_query, n_classes)

        return dists
