import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import*


class PoolLayer(nn.Module):
    def __init__(self):
        super(PoolLayer, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.pool_layer(x)

class Predictor(nn.Module):
    def __init__(self, in_features, num_classes, bottleneck_dim=1024, prob=0.5):
        super(Predictor, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.pool_layer(x)
        x = self.head(x)
        return x


class AdversarialNetwork(nn.Module):
    def __init__(self, in_features, bottleneck_dim=1024):
        super(AdversarialNetwork, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.ad_layer1 = nn.Linear(in_features, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool_layer(x)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x
