import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tqdm import trange
import scipy.sparse as sp

from config import Config
from SpatialDG.model import Spatial_GraST_DGI
from utils import (
    features_construct_graph, spatial_construct_graph1,
    normalize_sparse_matrix, sparse_mx_to_torch_sparse_tensor,
    ZINB, regularization_loss, consistency_loss, clustering
)


PLOT_COLOR = [
    "#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
    "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
    "#8B658B","#1E90FF","#AF5F3C","#CAFF70","#F9BD3F","#DAB370",
    "#877F6C","#268785","#82EF2D","#B4EEB4","#FF69B4"
]

def load_data(dataset_path, k=10, radius=150):
    """Load h5ad and build dual graphs."""
    adata = sc.read_h5ad(dataset_path)
    adata.var_names_make_unique()

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    features = torch.FloatTensor(X)

    # 构图
    fadj = features_construct_graph(features.numpy(), k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    # 归一化 & 转 torch.sparse
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return adata, features, nsadj, nfadj, graph_nei, graph_neg

def train_with_dgi(model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer):
    """单轮训练，返回嵌入与各损失。"""
    model.train()
    optimizer.zero_grad()

    forward_result = model(features, sadj, fadj, return_contrastive=True)
    if len(forward_result) == 7:
        com1, com2, emb, pi, disp, mean, dgi_loss = forward_result
    else:
        com1, com2, emb, pi, disp, mean = forward_result
        dgi_loss = torch.tensor(0.0, device=features.device)

    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)

    total_loss = (config.alpha * zinb_loss
                  + config.beta * con_loss
                  + config.gamma * reg_loss
                  + config.delta * dgi_loss)

    emb_np = pd.DataFrame(emb.detach().cpu().numpy()).fillna(0).values
    mean_np = pd.DataFrame(mean.detach().cpu().numpy()).fillna(0).values

    total_loss.backward()
    optimizer.step()

    return emb_np, mean_np, zinb_loss, reg_loss, con_loss, dgi_loss, total_loss

def cluster_and_visualize(adata, emb, n_clusters, palette=None, show=True):
    """聚类并绘制总览图。优先 mclust，失败回退 KMeans。"""
    adata.obsm["emb"] = emb
    palette = palette or PLOT_COLOR

    # 聚类
    try:
        clustering(adata, n_clusters, method="mclust")
    except Exception as e:
        print(f"mclust failed ({e}), fallback to KMeans...")
        adata.obs["domain"] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(emb)
    le = LabelEncoder()
    adata.obs["domain"] = le.fit_transform(adata.obs["domain"].astype(str))

    # 空间总览
    plt.rcParams["figure.figsize"] = (4, 4)
    ax = sc.pl.embedding(
        adata, basis="spatial", color="domain",
        s=30, show=False, palette=palette, title="Mouse Embryo E9.5 - SpatialDG"
    )
    ax.axis("off")
    if show:
        plt.show()

    return adata

def visualize(adata, palette=None, domains_to_show=None, flip_y=True, marker_genes=None, ncols=3):
    """可选每域可视化与 marker 基因可视化。"""
    palette = palette or PLOT_COLOR
    if flip_y and "spatial" in adata.obsm:
        adata.obsm["spatial"][:, 1] *= -1

    # 总览
    plt.rcParams["figure.figsize"] = (4, 4)
    ax = sc.pl.embedding(adata, basis="spatial",
                         color="domain", s=30, show=False, palette=palette,
                         title="Mouse Embryo E9.5 - SpatialDG")
    ax.axis("off")
    plt.show()

    # 单域高亮（其余灰色）
    if domains_to_show is not None:
        for domain_id in domains_to_show:
            col = f"domain_{domain_id}"
            adata.obs[col] = adata.obs["domain"].apply(lambda x: domain_id if x == domain_id else -1)
            custom_palette = {-1: "lightgray", domain_id: palette[domain_id % len(palette)]}
            sc.pl.embedding(adata, basis="spatial",
                            color=col, palette=custom_palette, s=30,
                            title=f"Domain {domain_id}", legend_loc="none", show=True)

    # marker 基因
    if marker_genes:
        genes = []
        for gs in marker_genes.values():
            for g in gs:
                if g in adata.var_names:
                    genes.append(g)
                else:
                    print(f"[warn] {g} not in adata.var_names")
        if genes:
            sc.pl.embedding(adata, basis="spatial", color=genes[:18], s=30,
                            ncols=ncols, cmap="Reds", use_raw=False, show=True)

def main(highly_genes=3000, save=False):
    dataset = "../data/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad"
    config_path = "./config/Mouse_Embryo.ini"
    save_path = "./result/Mouse_Embryo/"
    n_clusters = 22

    os.makedirs(save_path, exist_ok=True)
    config = Config(config_path)

    print("Loading data and building graphs...")
    adata, features, sadj, fadj, graph_nei, graph_neg = load_data(dataset)

    print("Initializing SpatialDG...")
    model = Spatial_GraST_DGI(
        nfeat=features.shape[1],
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout,
        dgi_sampling_strategy='adaptive',
        contrastive_dim=config.contrastive_dim,
        use_spatial_contrastive=config.use_spatial_contrastive
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print("Training...")
    for _ in trange(config.epochs):
        emb, mean, zinb_loss, reg_loss, con_loss, dgi_loss, total_loss = train_with_dgi(
            model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer
        )

    adata.obsm["spatial_dg_emb"] = emb

    print(f"Clustering into {n_clusters} domains...")
    adata = cluster_and_visualize(adata, emb, n_clusters, palette=PLOT_COLOR, show=True)


if __name__ == "__main__":
    main(highly_genes=3000, save=False)
