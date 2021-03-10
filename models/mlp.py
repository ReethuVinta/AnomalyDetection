import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, out_dim=14, in_channel=1, feature_sz=70, hidden_dim=256):
        super(MLP, self).__init__()
        self.in_dim = in_channel*feature_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Subject to be replaced dependent on task
        self.last = nn.Linear(hidden_dim, out_dim)

    def features(self, x):
        # x should be tensor here.
        # print('tpe of x',x)
        x = self.linear(x.float())
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def MLP100():
    return MLP(hidden_dim=100)


def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)
