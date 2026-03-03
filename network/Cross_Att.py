import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层 - 对应论文公式(1)-(4)
    功能：让EEG特征(query)去关注EM特征(key, value)，实现跨模态信息交互
    """
    def __init__(self, d_model):
        """
        Args:
            d_model: 特征维度（论文中是256）
        """
        super(CrossAttentionLayer, self).__init__()
        # 线性投影层：将输入特征映射到相同的维度空间
        # 对应论文公式(1)中的 W_Q, W_K, W_V
        self.linear_q = nn.Linear(d_model, d_model)  # 用于EEG的Query投影
        self.linear_k = nn.Linear(d_model, d_model)  # 用于EM的Key投影  
        self.linear_v = nn.Linear(d_model, d_model)  # 用于EM的Value投影

    def forward(self, x_q, x_kv):
        """
        前向传播
        Args:
            x_q: Query输入 (EEG特征) - 形状: (batch_size, d_model)
            x_kv: Key/Value输入 (EM特征) - 形状: (batch_size, d_model)
        Returns:
            cross_attended_features: 交叉注意力后的特征 - 形状: (batch_size, d_model)
        """
        # 1. 生成Q、K、V（对应论文公式1）
        Q = self.linear_q(x_q)  # (batch_size, d_model) - EEG作为Query
        K = self.linear_k(x_kv)  # (batch_size, d_model) - EM作为Key
        V = self.linear_v(x_kv)  # (batch_size, d_model) - EM作为Value

        # 2. 计算注意力权重（对应论文公式4中的softmax部分）
        # torch.matmul(Q, K.transpose(-2, -1)): Q和K的点积，计算相似度
        # math.sqrt(Q.size(-1)): 缩放因子，防止梯度消失
        # 最终形状: (batch_size, 1, 1) 因为这里是向量而非序列
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化为概率权重

        # 3. 加权融合（对应论文公式4中的加权求和）
        # 用注意力权重对Value进行加权，得到EEG关注的EM特征
        cross_attended_features = torch.matmul(attention_weights, V)  # (batch_size, d_model)

        return cross_attended_features


class MLCrossAttentionGating(nn.Module):
    """
    多层交叉注意力门控模块 - 对应论文的整个DIA层（公式1-5）
    包含：多层交叉注意力 + 门控机制 + 残差连接 + MLP融合
    论文中使用了3层(num_layers=3)
    """
    def __init__(self, eeg_dim=256, et_dim=177, d_model=256, num_layers=3):
        """
        Args:
            eeg_dim: EEG输入特征维度（论文中应该是256）
            et_dim: EM输入特征维度（论文中EM提取33个特征？这里是177，可能经过处理）
            d_model: 模型内部特征维度（论文中统一用256）
            num_layers: DIA层数（论文中使用3层，见图3分析）
        """
        super(MLCrossAttentionGating, self).__init__()
        self.d_model = d_model

        # 特征映射层：将不同模态的特征统一到相同维度
        # 注意：代码中注释掉了EEG的线性变换，可能EEG已经是d_model维度
        self.linear_eeg = nn.Linear(eeg_dim, d_model)  # EEG维度映射
        self.linear_et = nn.Linear(et_dim, d_model)    # EM维度映射

        # 门控机制线性层 - 对应论文中的Gate Attention (GA)机制
        # 用于计算每个通道的权重 w = sigmoid(W_GA * Z)
        self.gate = nn.Linear(d_model, d_model)

        # 创建多层交叉注意力（论文中分析1-4层效果，最终用3层）
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model) for _ in range(num_layers)
        ])

        # 融合MLP - 对应论文公式5中的MLP
        # 输入：拼接后的[原始EEG, 门控后的特征] (2*d_model)
        # 输出：融合特征 (d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),  # 降维
            nn.ReLU(),                          # 非线性激活
            nn.Linear(d_model, d_model)         # 保持维度
        )

    def forward(self, eeg_input, et_input):
        """
        前向传播 - 对应论文图1中的HAIA模块
        Args:
            eeg_input: EEG输入特征 - 形状: (batch_size, eeg_dim)
            et_input: EM输入特征 - 形状: (batch_size, et_dim)
        Returns:
            intermediate_outputs: 列表，包含每一层DIA的输出
                                每个元素形状: (batch_size, d_model)
        """
        # 1. 特征映射 - 对应论文中的 H_G 和 H_M
        # 将输入特征映射到统一的d_model维度
        eeg_transformed = eeg_input  # 注意：这里跳过了linear_eeg，可能输入已经是256维
        # eeg_transformed = self.linear_eeg(eeg_input)  # 被注释掉的原始代码
        et_transformed = self.linear_et(et_input)  # (batch_size, d_model)

        residual_output = eeg_transformed  # 初始化残差连接的基准值

        # 保存每一层的输出，用于后续的HRDA模块（多层MMD损失）
        intermediate_outputs = []

        # 2. 多层DIA处理（论文中使用3层）
        for layer_idx, layer in enumerate(self.cross_attention_layers):
            # 2.1 交叉注意力 - 对应公式(1)-(4)
            # unsqueeze(1)添加序列维度，因为CrossAttentionLayer设计为处理序列
            # 但这里实际上是向量，所以维度是(batch_size, 1, d_model)
            cross_attended_features = layer(
                residual_output.unsqueeze(1),  # EEG作为Query
                et_transformed.unsqueeze(1)     # EM作为Key/Value
            )  # 输出: (batch_size, 1, d_model)

            # 2.2 门控机制 - 对应论文中的Gate Attention
            # 计算门控权重 w = sigmoid(W_GA * Z)
            gate = torch.sigmoid(
                self.gate(cross_attended_features.squeeze(1))
            )  # (batch_size, d_model) - 每个通道的权重

            # 2.3 应用门控 - 对应公式中的 Z_GA = w ⊙ Z
            gated = cross_attended_features.squeeze(1) * gate  # (batch_size, d_model)

            # 2.4 特征拼接 - 对应公式5中的拼接操作 ⊕
            # 拼接原始EEG特征和门控后的特征
            combined = torch.cat((residual_output, gated), dim=1)  # (batch_size, 2*d_model)

            # 2.5 MLP融合 - 对应公式5中的MLP
            fused_output = self.fusion_mlp(combined)  # (batch_size, d_model)

            # 2.6 残差连接 - 对应公式5中的 F_final = F_fused + H_G
            # 将融合后的特征与原始EEG特征相加
            residual_output = eeg_transformed + fused_output  # (batch_size, d_model)

            # 保存当前层的输出，用于HRDA模块计算多层MMD损失
            intermediate_outputs.append(residual_output)

        # 返回所有中间层的输出
        # 注意：论文的HRDA模块需要使用这些中间特征计算多层MMD损失
        return intermediate_outputs