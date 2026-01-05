import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import os
import time
import torch
import scipy.sparse as sp

from config import Config
from model import Spatial_GraST_DGI
from utils import (
    spatial_construct_graph1, features_construct_graph,
    normalize_sparse_matrix, sparse_mx_to_torch_sparse_tensor,
    ZINB, regularization_loss, consistency_loss, clustering
)

N_TOP_GENES = 3000
N_EPOCHS = 200

input_dir = '../Data'
counts_file = os.path.join(input_dir, 'Puck_200127_15.digital_expression.txt')
coor_file = os.path.join(input_dir, 'Puck_200127_15_bead_locations.csv')

counts = pd.read_csv(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file)
coor_df = coor_df.set_index('barcode')

print(f"Expression matrix: {counts.shape}, Coordinates: {coor_df.shape}")

adata = sc.AnnData(counts.T)
adata.var_names_make_unique()

common_barcodes = adata.obs_names.intersection(coor_df.index)
print(f"Common barcodes: {len(common_barcodes)}")

adata = adata[common_barcodes, :]
coor_df = coor_df.loc[common_barcodes, ['xcoord', 'ycoord']]
adata.obsm["spatial"] = coor_df.to_numpy()

sc.pp.calculate_qc_metrics(adata, inplace=True)

used_barcode = pd.read_csv('../Data/used_barcodes.txt', sep='\t', header=None)
used_barcode = used_barcode[0]
valid_barcodes = adata.obs_names.intersection(used_barcode)
adata = adata[valid_barcodes, :]

sc.pp.filter_genes(adata, min_cells=50)
print('After filtering: ', adata.shape)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=N_TOP_GENES)
adata = adata[:, adata.var['highly_variable']].copy()
print('After HVG selection: ', adata.shape)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print('After normalization: ', adata.shape)

print("Building spatial and feature graphs...")
sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=50)
print(f"Spatial graph: {sadj.nnz} edges ({sadj.nnz/adata.shape[0]:.2f} edges/cell)")

X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
fadj = features_construct_graph(X, k=15)
print(f"Feature graph: {fadj.nnz} edges ({fadj.nnz/adata.shape[0]:.2f} edges/cell)")

print("Preparing training data...")

device = torch.device('cpu')

features = torch.FloatTensor(X).to(device)
n_input = features.shape[1]

nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
nfadj = sparse_mx_to_torch_sparse_tensor(nfadj).to(device)
nsadj = sparse_mx_to_torch_sparse_tensor(nsadj).to(device)

graph_nei = graph_nei.to(device)
graph_neg = graph_neg.to(device)

config = Config('config/MERFISH.ini')
config.epochs = N_EPOCHS

model = Spatial_GraST_DGI(
    nfeat=n_input,
    nhid1=config.nhid1,
    nhid2=config.nhid2,
    dropout=config.dropout,
    contrastive_dim=config.contrastive_dim,
    use_spatial_contrastive=config.use_spatial_contrastive
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

print(f"Model: {n_input} -> {config.nhid1} -> {config.nhid2}")
print(f"Dropout: {config.dropout}, LR: {config.lr}, Epochs: {config.epochs}")

print("\n" + "="*60)
print("Training started")
print("="*60)

model.train()
best_loss = float('inf')
start_time = time.time()

for epoch in range(config.epochs):
    epoch_start = time.time()
    optimizer.zero_grad()
    
    forward_result = model(features, nsadj, nfadj, return_contrastive=True)
    
    if len(forward_result) == 7:
        com1, com2, emb, pi, disp, mean, dgi_loss = forward_result
    else:
        com1, com2, emb, pi, disp, mean = forward_result
        dgi_loss = torch.tensor(0.0, device=device)
    
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    
    total_loss = (config.alpha * zinb_loss + 
                  config.beta * con_loss + 
                  config.gamma * reg_loss + 
                  config.delta * dgi_loss)
    
    total_loss.backward()
    optimizer.step()
    
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
    
    epoch_time = time.time() - epoch_start
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (config.epochs - epoch - 1)
        print(f"Epoch {epoch+1:3d}/{config.epochs} [{epoch_time:.1f}s] "
              f"Loss={total_loss.item():.4f} | ETA: {eta/60:.1f}min")

print(f"\nTraining completed. Best loss: {best_loss:.4f}, Time: {(time.time()-start_time)/60:.1f}min")

print("\nExtracting embeddings...")

model.eval()
with torch.no_grad():
    forward_result = model(features, nsadj, nfadj, return_contrastive=False)
    if len(forward_result) == 6:
        _, _, emb, pi, disp, mean = forward_result
    else:
        _, _, emb, pi, disp, mean, _ = forward_result
    
    embedding = emb.cpu().numpy()

adata.obsm['X_spatialdg'] = embedding
adata.obsm['emb'] = embedding
print(f"Embedding shape: {embedding.shape}")

print("\nClustering...")

from sklearn.cluster import KMeans

n_clusters = config.n_clusters if hasattr(config, 'n_clusters') else 11
print(f"Number of clusters: {n_clusters}")

try:
    clustering(adata, n_clusters, method='mclust')
    print("Clustering method: mclust")
except Exception as e:
    print(f"mclust failed ({e}), using KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    adata.obs['domain'] = kmeans.fit_predict(embedding).astype(str)

os.makedirs('output', exist_ok=True)
adata.write('output/spatialdg_result.h5ad')
print("Result saved to: output/spatialdg_result.h5ad")
