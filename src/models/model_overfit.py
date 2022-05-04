import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, BatchNorm2d

"""
16, 8 hidden shape

loss = 0.694008469581604: : 952it [10:22,  1.53it/s] 
Train value counts 0    229925
1     13787
dtype: int64
Train recall 0.058220786035849424
Train precision 0.15289765721331688
Train accuracy: 0.8122
EPOCH LOSS: 0.6931448262028334
loss = inf: 228it [02:31,  1.51it/s]
/datasets/anaconda3/envs/torch1.9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Value counts 0    58164
dtype: int64
Recall 0.0
Precision 0.0
Accuracy: 0.8532
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
        self.conv1 = GCNConv(input_dims, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        self.lin = Linear(
            hidden_channels[1]*nodes_per_graph,
            output_dim
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.conv1(x, edge_index)
        x = ReLU(self.hidden_channels[0]*self.nodes_per_graph)(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training, p=self.dropout)
        #x = F.dropout(x, training=self.training, p=self.dropout)
        x = torch.reshape(
            x,
            (
                -1,
                self.hidden_channels[1]*self.nodes_per_graph
            )
        )
        x = ReLU(self.hidden_channels*self.nodes_per_graph)(x)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
