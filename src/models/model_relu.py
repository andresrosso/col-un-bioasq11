import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, BatchNorm2d
"""
hidden layer 16

loss = 0.6921246647834778: : 952it [09:09,  1.73it/s]
Train value counts 1    190005
0     53707
dtype: int64
Train recall 0.7754025464689148
Train precision 0.14775926949290807
Train accuracy: 0.3022
EPOCH LOSS: 0.6933588130133492
loss = inf: 228it [02:46,  1.37it/s]
Value counts 1    48456
0     9708
dtype: int64
Recall 0.8324938502986998
Precision 0.1466691431401684
Accuracy: 0.2645
"""
class GraphClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        nodes_per_graph,
        hidden_channels,
        output_dim,
        dropout
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.nodes_per_graph = nodes_per_graph
        self.conv1 = GCNConv(input_dims, hidden_channels)
        self.lin = Linear(
            hidden_channels*nodes_per_graph,
            output_dim
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.conv1(x, edge_index)
        x = ReLU(self.hidden_channels*self.nodes_per_graph)(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        #x = F.dropout(x, training=self.training, p=self.dropout)
        x = torch.reshape(
            x,
            (
                -1,
                self.hidden_channels*self.nodes_per_graph
            )
        )
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
