import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GraphConvNN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, num_classes)

    def forward(self, data):
        data = Data(x=data[0], edge_index=data[1].t().contiguous())
        x, edge_index = data.x, data.edge_index


        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
