import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN

from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        pos_dim = 3
        feature_dim = 4 # 3 + 1 xgboost score
        sa1_dim = 64
        sa2_dim = 128
        sa3_dim = 256
        sa4_dim = 256
        self.cut_point = 0.00394
        
        self.sa1_module = SAModule(0.5, 7.0, MLP([pos_dim+feature_dim, sa1_dim, sa1_dim, sa2_dim]))

        self.sa2_module = SAModule(0.4, 16.0, MLP([pos_dim + sa2_dim , sa2_dim, sa2_dim, sa3_dim]))
        
        self.sa3_module = SAModule(0.4, 30.0, MLP([pos_dim + sa3_dim, sa3_dim, sa3_dim, sa4_dim]))
        
        self.sa4_module = GlobalSAModule(MLP([pos_dim + sa4_dim, sa4_dim, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + sa4_dim, sa4_dim, sa4_dim]))
        self.fp2_module = FPModule(3, MLP([sa4_dim + sa3_dim, sa4_dim, sa3_dim]))
        self.fp1_module = FPModule(3, MLP([sa3_dim + sa2_dim, sa3_dim, sa2_dim]))
        self.fp0_module = FPModule(3, MLP([sa2_dim + feature_dim, sa2_dim, 64]))

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        
        mask = data.xgboost_score > self.cut_point
        sa0_out = (torch.cat([data.x[mask], data.xgboost_score[mask, None]], dim=1) , data.pos[mask], data.batch[mask])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp3_out = self.fp3_module(*sa4_out, *sa3_out)
        fp2_out = self.fp2_module(*fp3_out, *sa2_out)
        fp1_out = self.fp1_module(*fp2_out, *sa1_out)
        x, _, _ = self.fp0_module(*fp1_out, *sa0_out)
        
        

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        out = torch.zeros_like(mask, dtype=torch.float)
        out[mask] = x.squeeze()
        del(x)
        out = self.sigmoid(out)
        
        return out

