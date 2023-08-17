import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, LayerNorm, GATConv
from torch_geometric.nn import global_mean_pool as gap

# GCNNet model
class GCNNet(torch.nn.Module):
    def __init__(self, num_features_xd=1280, hidden_dim=2048, output_dim=256, pre_dim=128, n_output=1, dropout=0.2):
        superGCNNetself).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(1280, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # self.cnn1 = nn.Conv1d(in_channels=num_features_xd, out_channels=hidden_dim, kernel_size=1)

        self.gat1 = GATConv(91, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)

        self.gat2 = GATConv(hidden_dim, output_dim)
        self.ln2 = LayerNorm(output_dim)

        self.dropout = dropout

        self.pre = nn.Sequential(
            nn.Linear(output_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pre_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pre_dim, n_output)
        )

    def forward(self, data):
        x1, edge_index, batch, global_features = data.x, data.edge_index, data.batch, data.global_features
        x1 = F.normalize(x1, p=2, dim=1)
        #  BLOSUM62 + PSSM + HMM + SPIDER3 + AAPHY7
        #  20 + 20 + 30 + 14 + 7 = 91
        # 1280 + 91 = 1371

        res = x1
        
        #91 维
        # gcn1
        x1 = torch.cat([x1[:, 1280:], x1[:, 1364:1364]], dim=1)
       

        x1 = self.gat1(x1, edge_index)
        x1 = self.ln1(x1, batch)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # liner1
        res1 = self.l1(res[:, :1280])

        # gcn2
        x1 = self.gat2(x1 + res1, edge_index)
        x1 = self.ln2(x1, batch)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)  # 256

        # liner2
        res1 = self.l2(res1)

        # readout
        x1 = gap(x1 + res1, batch)  # 256

        x1 = torch.cat([x1, global_features], -1)  # 256 + 10

        # 分类器
        out = self.pre(x1)

        return out
