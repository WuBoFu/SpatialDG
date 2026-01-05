"""
SpatialDG: Deep Graph Infomax for Spatial Transcriptomics

This module implements the core SpatialDG model that combines:
- Dual-view GCN (spatial and feature graphs)
- DGI contrastive learning
- ZINB reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SGC, GraphConvolution


class Discriminator(nn.Module):
    """Bilinear discriminator for DGI contrastive learning."""
    
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
    """Global graph readout via average pooling."""
    
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
    """
    Deep Graph Infomax contrastive loss.
    
    Supports two negative sampling strategies:
    - 'random': Random permutation (default, fast)
    - 'adaptive': Graph-aware hard negative sampling (slower, considers graph structure)
    """
    
    def __init__(self, hidden_dim, sampling_strategy='random'):
        super().__init__()
        self.discriminator = Discriminator(hidden_dim)
        self.readout = AvgReadout()
        self.sampling_strategy = sampling_strategy
        
    def _adaptive_negative_sampling(self, embeddings, A_s=None, A_f=None):
        """
        Adaptive negative sampling considering spatial and feature graph structure.
        
        Selects hard negatives that are dissimilar in embedding space but not 
        neighbors in either graph, providing more challenging contrastive pairs.
        """
        N = embeddings.size(0)
        device = embeddings.device
        
        if A_s is None and A_f is None:
            return embeddings[torch.randperm(N, device=device)]
        
        negative_indices = torch.zeros(N, dtype=torch.long, device=device)
        
        # Convert sparse matrices to dense if needed
        A_s_dense = A_s.to_dense() if A_s is not None and A_s.is_sparse else A_s
        A_f_dense = A_f.to_dense() if A_f is not None and A_f.is_sparse else A_f
        
        # Compute embedding similarity for hard negative selection
        emb_sim = torch.mm(embeddings, embeddings.t())
        emb_sim_norm = emb_sim / (torch.norm(embeddings, dim=1, keepdim=True) @ 
                                   torch.norm(embeddings, dim=1, keepdim=True).t() + 1e-8)
        
        for i in range(N):
            # Build candidate set (exclude self and neighbors)
            candidates_mask = torch.ones(N, dtype=torch.bool, device=device)
            candidates_mask[i] = False
            
            if A_s_dense is not None:
                candidates_mask &= ~(A_s_dense[i] > 0)
            if A_f_dense is not None:
                candidates_mask &= ~(A_f_dense[i] > 0)
            
            candidates = torch.where(candidates_mask)[0]
            
            if len(candidates) == 0:
                # Fallback to random if no valid candidates
                candidates = torch.randperm(N, device=device)
                candidates = candidates[candidates != i][:1]
                negative_indices[i] = candidates[0]
            else:
                # Select hard negative (highest similarity among non-neighbors)
                candidate_sims = emb_sim_norm[i, candidates]
                hardest_idx = torch.argmax(candidate_sims)
                negative_indices[i] = candidates[hardest_idx]
        
        return embeddings[negative_indices]
    
    def forward(self, embeddings, adj_mask=None, A_s=None, A_f=None):
        """
        Compute DGI contrastive loss.
        
        Args:
            embeddings: Node embeddings [N, hidden_dim]
            adj_mask: Adjacency mask for global representation (optional)
            A_s: Spatial adjacency matrix (for adaptive sampling)
            A_f: Feature adjacency matrix (for adaptive sampling)
        """
        batch_size = embeddings.size(0)
        
        # Global representation
        global_repr = self.readout(embeddings, adj_mask)
        
        # Generate negative samples
        if self.sampling_strategy == 'random':
            negative_embeddings = embeddings[torch.randperm(batch_size)]
        elif self.sampling_strategy == 'adaptive':
            negative_embeddings = self._adaptive_negative_sampling(embeddings, A_s, A_f)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Discriminator prediction
        logits = self.discriminator(global_repr, embeddings, negative_embeddings)
        
        # Binary cross-entropy loss
        pos_labels = torch.ones(batch_size, 1, device=embeddings.device)
        neg_labels = torch.zeros(batch_size, 1, device=embeddings.device)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1, 1), labels, reduction='mean'
        )
        return loss


class DGIDataAugmentation:
    """Data augmentation for DGI-style contrastive learning."""
    
    def __init__(self, corruption_ratio=0.2):
        self.corruption_ratio = corruption_ratio
    
    def feature_corruption(self, features):
        """Corrupt features by random permutation."""
        corrupted = features.clone()
        num_corrupt = int(features.size(0) * self.corruption_ratio)
        corrupt_indices = torch.randperm(features.size(0))[:num_corrupt]
        corrupted[corrupt_indices] = corrupted[corrupt_indices][torch.randperm(num_corrupt)]
        return corrupted
    
    def edge_corruption(self, adj):
        """Corrupt edges by random dropout."""
        adj_dense = adj.to_dense() if adj.is_sparse else adj.clone()
        mask = torch.rand_like(adj_dense) > self.corruption_ratio
        corrupted_adj = adj_dense * mask
        return corrupted_adj.to_sparse() if adj.is_sparse else corrupted_adj
    
    def create_corrupted_views(self, features, sadj, fadj):
        """Create corrupted views for contrastive learning."""
        return {
            'corrupted_features': self.feature_corruption(features),
            'corrupted_sadj': self.edge_corruption(sadj),
            'corrupted_fadj': self.edge_corruption(fadj)
        }


class Attention(nn.Module):
    """Multi-view attention for combining different graph embeddings."""
    
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
    """Two-layer Graph Convolutional Network."""
    
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    """ZINB decoder for gene expression reconstruction."""
    
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
    """
    SpatialDG: Spatial Transcriptomics with Deep Graph Infomax.
    
    Combines dual-view GCN (spatial and feature graphs) with DGI contrastive 
    learning and ZINB reconstruction for spatial transcriptomics analysis.
    
    Args:
        nfeat: Input feature dimension
        nhid1: Hidden layer 1 dimension  
        nhid2: Hidden layer 2 / output dimension
        dropout: Dropout rate
        use_spatial_contrastive: Whether to use spatial-aware contrastive learning
        dgi_sampling_strategy: 'random' or 'adaptive' negative sampling
    """
    
    def __init__(self, nfeat, nhid1, nhid2, dropout, contrastive_dim=64, 
                 use_spatial_contrastive=True, dgi_sampling_strategy='random'):
        super().__init__()
        
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)  # Spatial GCN
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)  # Feature GCN
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)  # Shared GCN
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(nn.Linear(nhid2, nhid2))
        
        self.use_spatial_contrastive = use_spatial_contrastive
        self.dgi_loss_fn = DGILoss(nhid2, sampling_strategy=dgi_sampling_strategy)
        self.dgi_sampling_strategy = dgi_sampling_strategy
        self.dropout = dropout
    
    def forward(self, x, sadj, fadj, return_contrastive=True):
        """
        Forward pass.
        
        Args:
            x: Input features [N, nfeat]
            sadj: Spatial adjacency matrix [N, N]
            fadj: Feature adjacency matrix [N, N]
            return_contrastive: Whether to compute and return DGI loss
            
        Returns:
            com1, com2: Shared GCN outputs for spatial/feature graphs
            emb: Final fused embedding
            pi, disp, mean: ZINB parameters
            dgi_loss: DGI contrastive loss (if return_contrastive=True and training)
        """
        # Dual-view GCN encoding
        emb1 = self.SGCN(x, sadj)  # Spatial view
        com1 = self.CGCN(x, sadj)  # Shared encoder on spatial graph
        com2 = self.CGCN(x, fadj)  # Shared encoder on feature graph
        emb2 = self.FGCN(x, fadj)  # Feature view

        # Multi-view attention fusion
        emb = torch.stack([emb1, (com1 + com2) / 2, emb2], dim=1)
        emb, att = self.att(emb)
        emb = self.MLP(emb)
       
        # ZINB reconstruction
        [pi, disp, mean] = self.ZINB(emb)
        
        if not return_contrastive or not self.training:
            return com1, com2, emb, pi, disp, mean
        
        # DGI contrastive learning
        if self.use_spatial_contrastive:
            adj_mask = sadj.to_dense() if sadj.is_sparse else sadj
            
            if self.dgi_sampling_strategy == 'adaptive':
                A_s = sadj.to_dense() if sadj.is_sparse else sadj
                A_f = fadj.to_dense() if fadj.is_sparse else fadj
                dgi_loss = self.dgi_loss_fn(emb, adj_mask=adj_mask, A_s=A_s, A_f=A_f)
            else:
                dgi_loss = self.dgi_loss_fn(emb, adj_mask=adj_mask)
        else:
            dgi_loss = self.dgi_loss_fn(emb)
                
        return com1, com2, emb, pi, disp, mean, dgi_loss

