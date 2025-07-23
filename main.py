import torch
from torch import nn
from torchvision import models


class PGUnit(nn.Module):
    """Patch-Gated单元模块，局部特征提取与加权"""
    def __init__(self, feat_dim, output_dim=128):
        super(PGUnit, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
        # 降维
        self.feature_transform = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """输入局部特征，输出加权特征与注意力权重"""
        attention_weight = self.attention_net(x)
        # 特征降维
        x = self.feature_transform(x)
        attention_weight = attention_weight.view(attention_weight.size(0), 1)
        weighted_feature = attention_weight * x
        return weighted_feature, attention_weight


class FDN(nn.Module):
    """局部特征提取与加权模块"""
    def __init__(self, num_branch, basic_dim, feat_dim):
        super(FDN, self).__init__()
        self.M = num_branch
        self.P = basic_dim
        self.D = feat_dim
        for i in range(self.M):
            setattr(self, f"fdn_fc{i}", PGUnit(self.P, self.D))  # 调整 PGUnit 的输出维度

    def forward(self, x):
        features, alphas = [], []
        for i in range(self.M):
            feature, alpha = getattr(self, f"fdn_fc{i}")(x)
            features.append(feature)
            alphas.append(alpha)

        features = torch.stack(features).permute([1, 0, 2])  # [batch_size, num_branch, feat_dim]
        alphas = torch.cat(alphas, dim=1)  # [batch_size, num_branch]
        return features, alphas


class FRN(nn.Module):
    """局部特征融合模块"""
    def __init__(self, num_branch, feat_dim):
        super(FRN, self).__init__()
        self.M = num_branch
        self.D = feat_dim
        self.intra_rm = nn.ModuleList([nn.Linear(self.D, self.D) for _ in range(self.M)])
        self.inter_rm = nn.ModuleList([nn.Linear(self.D, self.D) for _ in range(self.M)])
        self.delta = 0.5

    def forward(self, x, alphas):
        intra_features = [torch.tanh(self.intra_rm[i](x[:, i, :])) for i in range(self.M)]
        intra_features = torch.stack(intra_features).permute([1, 0, 2])  # [batch_size, num_branch, feat_dim]

        inter_features = [torch.relu(self.inter_rm[i](intra_features[:, i, :])) for i in range(self.M)]
        inter_features = torch.stack(inter_features).permute([1, 0, 2])
        inter_features = self.delta * x + (1 - self.delta) * inter_features
        return inter_features


class FDRL(nn.Module):
    """融合局部与全局特征的情感识别模型"""
    def __init__(self, num_branch, basic_dim, feat_dim, num_class):
        super(FDRL, self).__init__()
        self.M = num_branch
        self.P = basic_dim
        self.D = feat_dim
        self.num_class = num_class

        # 全局特征提取
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 局部特征提取与融合模块
        self.fdn = FDN(self.M, self.P, self.D)
        self.frn = FRN(self.M, self.D)

        # 调整输入维度
        self.fc_fusion = nn.Linear(self.D * self.M + self.P, 512)
        self.classifier = nn.Linear(512, self.num_class)

    def forward(self, x):
        """前向传播"""
        global_feat = self.backbone(x)
        global_feat = self.pooling(global_feat).view(global_feat.size(0), -1)  # [batch_size, 512]

        fdn_feat, alphas = self.fdn(global_feat)  # 局部特征提取
        frn_feat = self.frn(fdn_feat, alphas)  # 局部特征融合

        frn_feat = frn_feat.reshape(frn_feat.size(0), -1)  # [batch_size, num_branch * feat_dim]


        # 拼接局部和全局特征 维度 [batch_size, 1664]
        combined_features = torch.cat([frn_feat, global_feat], dim=1)
        combined_features = torch.relu(self.fc_fusion(combined_features))

        output = self.classifier(combined_features)
        return fdn_feat, alphas, output
