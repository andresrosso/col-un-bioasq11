import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, BatchNorm2d
"""
hidden layer 16

loss = 0.692568302154541: : 952it [08:05,  1.96it/s] 
Train value counts 1    202629
0     41083
dtype: int64
Train recall 0.8310547684149474
Train precision 0.14849799387057133
Train accuracy: 0.2669
EPOCH LOSS: 0.6933017681751933
loss = inf: 228it [02:45,  1.38it/s]
Value counts 1    48456
0     9708
dtype: int64
Recall 0.8324938502986998
Precision 0.1466691431401684
Accuracy: 0.2645
Value counts 1    49318
0     9199
dtype: int64
Recall 0.8353049270701735
Precision 0.1474715114157103
Accuracy: 0.2570
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
        #x = F.dropout(x, training=self.training, p=self.dropout)
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
