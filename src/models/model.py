import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm2d

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
