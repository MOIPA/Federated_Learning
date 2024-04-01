import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd

"""

在这个模型中,16是第一个图卷积层(GCNConv)的输出特征数量。在图神经网络中,每一层都会对节点的特征进行转换,
输出新的特征表示。这个数字决定了转换后的特征数量。

在你的模型中,self.conv1 = GCNConv(18, 16)表示第一个图卷积层接收每个节点的18个特征,并输出16个新的特征。
这16个特征然后被用作下一层(self.conv')的输入。

这个数字可以根据你的具体任务和数据进行调整。更多的特征可能会帮助模型捕捉更复杂的模式,但也可能导致过拟合。
同样,更少的特征可能使模型更简单,但可能无法捕捉到所有重要的信息。这是一个需要通过实验来找到最优值的超参数
"""


# Define a simple GNN with GCN layers
class Net18(torch.nn.Module):
    def __init__(self, num_output_features):
        super(Net18, self).__init__()
        self.conv1 = GCNConv(18, 16)
        self.conv2 = GCNConv(16, num_output_features)  # Binary classification
        self.num_output_features = num_output_features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
