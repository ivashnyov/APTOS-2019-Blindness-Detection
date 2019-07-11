import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceEmbedder(nn.Module):
    def __init__(self, original_model, num_embed=128):
        super().__init__()

        self.model_name = 'embedder'
        self.num_embed = num_embed

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.embedding = nn.Linear(2048, num_embed)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embed = self.embedding(features)
        return embed


class CosThetaLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        w = self.weight
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = w.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(w)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta


class FaceNet(nn.Module):
    def __init__(self, embedder, num_classes=8630):
        super().__init__()

        self.model_name = 'facenet'
        self.num_classes = num_classes

        if not isinstance(embedder, FaceEmbedder):
            raise ValueError('original_model should be instance of FaceEmbedder instead of {}'.format(embedder))
        self.embedder = embedder

        self.fc_prelu = nn.PReLU()
        self.fc_theta = CosThetaLinear(embedder.num_embed, num_classes)

    def forward(self, x):
        embed = self.embedder(x)
        out = self.fc_theta(self.fc_prelu(embed))
        return out


# --- sphereface https://arxiv.org/abs/1704.08063 ---
class SphereLoss(nn.Module):
    def __init__(self, s=64, m=4):
        super(SphereLoss, self).__init__()
        self.s = s
        self.m = m
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, cos_theta, target):
        self.it += 1
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.acos()
        k = (self.m * theta / 3.14159265).floor()
        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * cos_m_theta - 2 * k

        target = target.view(-1, 1)
        index = torch.zeros_like(cos_theta)  # size=(B,Classnum)
        index.scatter_(1, target, 1)
        index = index.byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.001 * self.it))
        output = self.s * cos_theta  # size=(B,Classnum)
        output[index] -= self.s * cos_theta[index] * 1.0 / (1 + self.lamb)
        output[index] += self.s * phi_theta[index] * 1.0 / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        loss = -logpt.mean()

        return loss


# -- arcface https://arxiv.org/abs/1801.09414 ---
class ArcLoss(nn.Module):
    def __init__(self, s=64, m=0.35):
        super(ArcLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)

    def forward(self, cos_theta, target):
        cos_theta2 = cos_theta ** 2
        sin_theta2 = 1 - cos_theta2
        sin_theta = torch.sqrt(sin_theta2).clamp(0, 1)

        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = cos_theta_m.clamp(-1, 1)

        # cos(theta + m) or (cos(theta) - m * sin(m))
        phi_theta = cos_theta - self.m * self.sin_m
        cond = cos_theta > self.threshold
        cos_theta_m = torch.where(cond, cos_theta_m, phi_theta)

        target = target.view(-1, 1)
        index = torch.zeros_like(cos_theta)  # size=(B,Classnum)
        index.scatter_(1, target, 1)
        index = index.byte()
        output = self.s * cos_theta  # size=(B,Classnum)
        output[index] = self.s * cos_theta_m[index]

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        loss = -logpt.mean()

        return loss


# -- cosineface https://arxiv.org/abs/1801.07698.pdf ---
class CosineLoss(nn.Module):
    def __init__(self, s=64, m=0.15):
        super(CosineLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cos_theta, target):
        cos_theta_m = cos_theta - self.m

        target = target.view(-1, 1)
        index = torch.zeros_like(cos_theta)  # size=(B,Classnum)
        index.scatter_(1, target, 1)
        index = index.byte()
        output = self.s * cos_theta  # size=(B,Classnum)
        output[index] = self.s * cos_theta_m[index]

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        loss = -logpt.mean()

        return loss


# -- insightface cos(m1 * theta + m2) - m3 ---
class InsightLoss(nn.Module):
    def __init__(self, s=64, m1=0.9, m2=0.4, m3=0.15):
        super(InsightLoss, self).__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.cos_m2 = math.cos(m2)
        self.sin_m2 = math.sin(m2)
        self.threshold = math.cos((math.pi - m2) / m1)

    def forward(self, cos_theta, target):
        # theta = torch.acos(cos_theta)
        # theta_m = self.m1 * theta + self.m2
        # cos_theta_m = torch.cos(theta_m).clamp(-1, 1)

        cos_theta2 = cos_theta ** 2
        sin_theta2 = 1 - cos_theta2
        sin_theta = torch.sqrt(sin_theta2).clamp(0, 1)

        cos_theta_m = cos_theta * self.cos_m2 - sin_theta * self.sin_m2
        cos_theta_m = cos_theta_m.clamp(-1, 1)

        phi_theta = cos_theta - self.m2 * self.sin_m2
        cond = cos_theta > self.threshold
        cos_theta_m = torch.where(cond, cos_theta_m, phi_theta)

        cos_theta_m = cos_theta_m - self.m3

        target = target.view(-1, 1)
        index = torch.zeros_like(cos_theta)  # size=(B,Classnum)
        index.scatter_(1, target, 1)
        index = index.byte()
        output = self.s * cos_theta  # size=(B,Classnum)
        output[index] = self.s * cos_theta_m[index]

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        loss = -logpt.mean()

        return loss