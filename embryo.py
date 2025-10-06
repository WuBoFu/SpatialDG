# run_mouse_embryo.py
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import os
from config import Config

device = torch.device('mps')
os.environ["LD_LIBRARY_PATH"] = f"{os.popen('python -m rpy2.situation LD_LIBRARY_PATH').read().strip()}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Import your model and utilities
from SpatialDG.model import Spatial_GraST_DGI
from utils import *

# Color palette similar to GraphST
PLOT_COLOR = ["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
              "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
              "#8B658B","#1E90FF","#AF5F3C","#CAFF70","#F9BD3F","#DAB370",
              "#877F6C","#268785",'#82EF2D','#B4EEB4']

def load_data(dataset_path):
    """Load Mouse Embryo data"""
    adata = sc.read_h5ad(dataset_path)
    adata.var_names_make_unique()

    # Generate data if not preprocessed
    features = torch.FloatTensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    
    # Construct graphs
    print("Constructing graphs...")
    fadj = features_construct_graph(features.numpy(), k=10)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=150)
    
    # Normalize adjacency matrices
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    
    return adata, features, nsadj, nfadj, graph_nei, graph_neg

from tqdm import trange

def train_with_dgi(model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer):
    """DGI训练函数"""
    model.train()
    optimizer.zero_grad()
    forward_result = model(features, sadj, fadj, return_contrastive=True)
    
    if len(forward_result) == 7:  # 包含DGI损失
        com1, com2, emb, pi, disp, mean, dgi_loss = forward_result
    else:  
        com1, com2, emb, pi, disp, mean = forward_result
        dgi_loss = torch.tensor(0.0)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = (config.alpha * zinb_loss + 
                  config.beta * con_loss + 
                  config.gamma * reg_loss +
                  config.delta * dgi_loss)  # DGI损失权重
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    
    total_loss.backward()
    optimizer.step()
    
    return emb, mean, zinb_loss, reg_loss, con_loss, dgi_loss, total_loss

def cluster_and_visualize(adata, emb, n_clusters, save_path, dataset_name='Mouse_Embryo', plot_color=None):
    """Perform clustering and create visualizations"""
    # Clustering using mclust or KMeans
    from utils import mclust_R
    
    adata.obsm['emb'] = emb
    
    # Try mclust first, fallback to KMeans
    try:
        # clustering without refinement
        clustering(adata, n_clusters, method='mclust')
        # adata.obs['domain_mclust_no_refine'] = adata.obs['domain']

    except:
        print("mclust failed, using KMeans...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # adata.obs['domain'] = kmeans.fit_predict(emb)
   
    # Create save directory
    # os.makedirs(save_path, exist_ok=True)
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # adata.obs['domain'] = le.fit_transform(adata.obs['domain'].astype(str))

    # 1. Overall visualization
    plt.rcParams["figure.figsize"] = (3, 4)
    plot_color=["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
            "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
            "#8B658B","#1E90FF","#AF5F3C","#CAFF70", "#F9BD3F","#DAB370",
            "#877F6C","#268785", '#82EF2D', '#B4EEB4']

    ax = sc.pl.embedding(adata, basis="spatial",
                        color="domain",
                        s=30,
                        show=False,
                        palette=plot_color,
                        title='GraphST')
    domains_to_show = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16, 17, 18, 19, 20, 21, 22] 
    for i, domain_id in enumerate(domains_to_show):
    # 创建一个新的列，只显示特定域
        adata.obs[f'domain_{domain_id}'] = adata.obs['domain'].apply(
        lambda x: domain_id if x == domain_id else -1  # 使用-1表示其他域
    )
    # 创建自定义调色板：-1为灰色，domain_id为原始颜色
    custom_palette = {-1: 'lightgray', domain_id: plot_color[domain_id % len(plot_color)]}


    plt.rcParams["figure.figsize"] = (3, 4)
    sc.pl.embedding(adata, basis="spatial",
                    color=f'domain_{domain_id}',
                    palette=custom_palette,
                    s=30,
                    title=f'Domain {domain_id}',
                    legend_loc='none',
                    show=True)

    
    # plt.savefig(os.path.join(save_path, 'spatial_domains.png'), 
    #             dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Individual domain visualization (optional)
    
    # if n_clusters <= 30:  # Only for reasonable number of clusters
    #     visualize_individual_domains(adata, n_clusters, save_path, plot_color)
    domains_to_show = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16, 17, 18, 19, 20, 21, 22]  # 例如：显示第1、2、4、5、10、12、15个域

# 为每个域创建单独的可视化，使用相同的颜色
    for i, domain_id in enumerate(domains_to_show):
        # 创建一个新的列，只显示特定域
        adata.obs[f'domain_{domain_id}'] = adata.obs['domain'].apply(
            lambda x: domain_id if x == domain_id else -1  # 使用-1表示其他域
        )
        
        # 创建自定义调色板：-1为灰色，domain_id为原始颜色
        custom_palette = {-1: 'lightgray', domain_id: plot_color[domain_id % len(plot_color)]}
        
        # 绘制特定域
        sc.pl.embedding(adata, basis="spatial",
                        color=f'domain_{domain_id}',
                        palette=custom_palette,
                        s=30,
                        title=f'Domain {domain_id}',
                        legend_loc='none',
                        show=True)
        plt.tight_layout()
        plt.show()

    return adata

def visualize_individual_domains(adata, n_clusters, save_path):
    n_show = min(4, n_clusters)
    domains_to_show = np.linspace(0, n_clusters-1, n_show, dtype=int)
    plot_color=["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
           "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
           "#8B658B","#1E90FF","#AF5F3C","#CAFF70", "#F9BD3F","#DAB370",
          "#877F6C","#268785", '#82EF2D', '#B4EEB4']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, domain_id in enumerate(domains_to_show):
        if idx < len(axes):
            col_name = f'highlight_domain_{domain_id}'
            adata.obs[col_name] = adata.obs['domain'].apply(lambda x: domain_id if x == domain_id else -1)
            custom_palette = {-1: 'lightgray', domain_id: plot_color[domain_id % len(plot_color)]}

            sc.pl.embedding(
                adata,
                basis="spatial",
                color="domain",
                palette=plot_color,
                s=30,
                ax=axes[idx],
                title=f'Domain {domain_id}',
                legend_loc='none',
                show=False
            )
            axes[idx].axis('off')

    for idx in range(len(domains_to_show), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'individual_domains.png'), dpi=300, bbox_inches='tight')
    plt.show()
def visualize(adata, plot_color=None, domains_to_show=None, marker_genes=None):
    import matplotlib.pyplot as plt
    import scanpy as sc

    # 翻转y轴
    adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    plt.rcParams["figure.figsize"] = (4, 4)
    # 主 embedding 可视化
    ax = sc.pl.embedding(adata, basis="spatial",
                         color="domain",
                         s=30,
                         show=False,
                         palette=plot_color,
                         title='Mouse Embryo E9.5 - Spatial-DG')
    ax.axis('off')
    ax.set_title('Mouse Embryo E9.5')

    # 可视化特定空间域
    if domains_to_show is not None:
        for i, domain_id in enumerate(domains_to_show):
            adata.obs[f'domain_{domain_id}'] = adata.obs['domain'].apply(
                lambda x: domain_id if x == domain_id else -1
            )
            custom_palette = {-1: 'lightgray', domain_id: plot_color[domain_id % len(plot_color)]}
            sc.pl.embedding(adata, basis="spatial",
                            color=f'domain_{domain_id}',
                            palette=plot_color,
                            s=30,
                            title=f'Domain {domain_id}',
                            legend_loc='none',
                            show=True)

    # 标记基因表达可视化
    if marker_genes is not None:
        available_genes = []
        for genes in marker_genes.values():
            for gene in genes:
                if gene in adata.var_names:
                    available_genes.append(gene)
                    print(f"Found {gene}")
                else:
                    print(f"{gene} not found in adata.var_names")
        if available_genes:
            sc.pl.embedding(adata, basis="spatial",
                            color=available_genes[0:18],
                            s=30,
                            ncols=3,
                            cmap='Reds',
                            use_raw=False,
                            show=True)
    print(adata.var_names)
    print(adata.var.head())
def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    # 保证 adata.X 是 np.ndarray
    if hasattr(adata.X, 'A'):
        adata.X = adata.X.A
    else:
        adata.X = np.array(adata.X)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata

def main(highly_genes):
    
    dataset = '../data/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad'
    config_path = './config/Mouse_Embryo.ini'
    save_path = './result/Mouse_Embryo/'
    n_clusters=22
    config = Config(config_path)
    adata, features, sadj, fadj, graph_nei, graph_neg = load_data(dataset)
    print("Loading data...")

    # Select highly variable genes
    adata.var_names_make_unique()
    # adata = normalize(adata, highly_genes=highly_genes)

    # Initialize model
    print("Initializing Spatial-DG model...")
    model = Spatial_GraST_DGI(
        nfeat=features.shape[1],
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout,
        contrastive_dim=config.contrastive_dim,
        use_spatial_contrastive=config.use_spatial_contrastive
    )
    
    # Move to GPU if available
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    for epoch in trange(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, dgi_loss, total_loss = train_with_dgi(
                model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer
            )
    
    # Save embeddings
    adata.obsm['spatial_dg_emb'] = emb
    # adata.layers['reconstructed'] = mean
    
    # Move features back to CPU for clustering
    features = features.cpu()
    
    # Cluster and visualize
    print(f"Clustering into {n_clusters} domains...")
    adata = cluster_and_visualize(adata, emb, n_clusters, 
                                 save_path, 'Mouse Embryo E9.5')
    # 用法示例
    os.makedirs(save_path, exist_ok=True)
    
    marker_genes = {
    "Cavity": ['Pecam1', 'Cdh5', 'Kdr'],  # 血管内皮标记
    "Mesenchyme": ['Meox1', 'Vim', 'Fn1'],     # 胶原蛋白1、波形蛋白、纤连蛋白
    "Cavity": ['Tie1', 'Flt1', 'Tek'],    # 血管发育标记    # Alb（白蛋白），Afp（甲胎蛋白），Hnf4a（肝细胞核因子4α）
    "Cavity":   ['Hbb-bt', 'Hba-a1', 'Hba-a2'],  # 血红蛋白基因，这组是正确的
    "Neural_tube": ['Krt5', 'Krt14', 'Krt15'],

    
}

    domains_to_show = list(range(0, 23))
    plot_color = ["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
                  "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
                  "#8B658B","#1E90FF","#AF5F3C","#CAFF70", "#F9BD3F","#DAB370",
                  "#877F6C","#268785", '#82EF2D', '#B4EEB4']
    visualize(adata, plot_color=plot_color, domains_to_show=domains_to_show, marker_genes=marker_genes)
    # Save results
    adata.write(os.path.join(save_path, 'spatial_dg_result.h5ad'))
    pd.DataFrame(emb).to_csv(os.path.join(save_path, 'embeddings.csv'))

    print(f"\nResults saved to {save_path}")
    print("Done!")

if __name__ == "__main__":
    main(highly_genes=3000)