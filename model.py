import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import SGC, GraphConvolution
import pandas as pd

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)  # positive sample
        sc_2 = self.f_k(h_mi, c_x)  # negative sample
        
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
            
        logits = torch.cat((sc_1, sc_2), 1)
        return logits

class AvgReadout(nn.Module):
    """global embedding"""
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        if mask is not None:
            vsum = torch.mm(mask, emb)
            row_sum = torch.sum(mask, 1)
            row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
            global_emb = vsum / row_sum
        else:
            global_emb = torch.mean(emb, dim=0, keepdim=True)
        return F.normalize(global_emb, p=2, dim=1)

class DGILoss(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.discriminator = Discriminator(hidden_dim)
        self.readout = AvgReadout()
        
    def forward(self, embeddings, adj_mask=None):
        """
        Args:
            embeddings: 节点嵌入 [N, hidden_dim]
            adj_mask: 邻接掩码（可选）
        """
        batch_size = embeddings.size(0)
        if adj_mask is not None:
            global_repr = self.readout(embeddings, adj_mask)
        else:
            global_repr = self.readout(embeddings)
        negative_embeddings = embeddings[torch.randperm(batch_size)]
        logits = self.discriminator(global_repr, embeddings, negative_embeddings)
        pos_labels = torch.ones(batch_size, 1, device=embeddings.device)
        neg_labels = torch.zeros(batch_size, 1, device=embeddings.device)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1, 1), 
            labels, 
            reduction='mean'
        )
        
        return loss

class DGIDataAugmentation:
    def __init__(self, corruption_ratio=0.2):
        self.corruption_ratio = corruption_ratio
    
    def feature_corruption(self, features):
        """特征腐败：随机排列部分特征"""
        corrupted = features.clone()
        num_corrupt = int(features.size(0) * self.corruption_ratio)
        corrupt_indices = torch.randperm(features.size(0))[:num_corrupt]
        
        # 随机排列选中的节点特征
        corrupted[corrupt_indices] = corrupted[corrupt_indices][torch.randperm(num_corrupt)]
        return corrupted
    
    def edge_corruption(self, adj):
        """边腐败：随机删除部分边"""
        if adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj.clone()
        mask = torch.rand_like(adj_dense) > self.corruption_ratio
        corrupted_adj = adj_dense * mask
        
        return corrupted_adj.to_sparse() if adj.is_sparse else corrupted_adj
    
    def create_corrupted_views(self, features, sadj, fadj):
        """创建腐败视图"""
        return {
            'corrupted_features': self.feature_corruption(features),
            'corrupted_sadj': self.edge_corruption(sadj),
            'corrupted_fadj': self.edge_corruption(fadj)
        }
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        # self.gc1 = GATv2(nfeat, nhid)
        # self.gc2 = GATv2(nhid, out)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
    
    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]
    
class Spatial_GraST_DGI(nn.Module):
    """使用DGI对比学习的空间转录组模型"""
    
    def __init__(self, nfeat, nhid1, nhid2, dropout, contrastive_dim=64, 
                 use_spatial_contrastive=True):
        super().__init__()
        
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(nn.Linear(nhid2, nhid2))
        self.use_spatial_contrastive = use_spatial_contrastive
        self.dgi_loss_fn = DGILoss(nhid2)
        self.dropout = dropout
    
    def forward(self, x, sadj, fadj, return_contrastive=True):
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        com1 = self.CGCN(x, sadj)  # Co_GCN
        com2 = self.CGCN(x, fadj)  # Co_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        emb = torch.stack([emb1, (com1 + com2) / 2, emb2], dim=1)
        emb, att = self.att(emb)
        emb = self.MLP(emb)
        [pi, disp, mean] = self.ZINB(emb)
        
        if not return_contrastive or not self.training:
            return com1, com2, emb, pi, disp, mean
        if self.use_spatial_contrastive:
            dgi_loss = self.dgi_loss_fn(emb, sadj.to_dense())
        else:
            dgi_loss = self.dgi_loss_fn(emb)       
        return com1, com2, emb, pi, disp, mean, dgi_loss

