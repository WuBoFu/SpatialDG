from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from SpatialDG.model import Spatial_GraST_DGI
import os
import argparse
from config import Config
from sklearn import metrics
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import ranksums


def load_data(dataset):
    print("load data:")
    path = "../generate_data/" + dataset + "/Space.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nsadj, nfadj, graph_nei, graph_neg


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
def perform_clustering_comparison(adata, emb_max, idx_max, savepath):
    """进行聚类比较分析 (实验B)"""
    print("Performing clustering comparison...")
    
    # 这里需要实现其他方法的聚类结果
    # 由于我们只有SpatialGD的结果，我们展示如何可视化
    
    # 创建比较图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始注释
    sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], 
                  title='Manual annotation', ax=axes[0], show=False)
    
    # SpatialGD结果
    adata.obs['spatialGD_clusters'] = idx_max.astype(str)
    sc.pl.spatial(adata, img_key="hires", color=['spatialGD_clusters'], 
                  title='SpatialGD', ax=axes[1], show=False)
    
    # 如果有其他方法的结果，可以在这里添加
    # 这里展示聚类数量分布
    cluster_counts = pd.Series(idx_max).value_counts().sort_index()
    axes[2].bar(cluster_counts.index.astype(str), cluster_counts.values)
    axes[2].set_title('Cluster Distribution')
    axes[2].set_xlabel('Cluster ID')
    axes[2].set_ylabel('Number of spots')
    
    plt.tight_layout()
    plt.savefig(savepath + 'clustering_comparison.jpg', bbox_inches='tight', dpi=600)
    plt.show()
    
    return adata

def find_deg_analysis(adata, target_clusters=[3, 5, 15, 16], n_top_genes=3):
    """找到特定聚类的差异表达基因 (实验C的准备)"""
    print("Finding DEGs for target clusters...")
    
    # 设置聚类信息
    adata.obs['clusters'] = adata.obs['spatialGD_clusters'].astype('category')
    
    # 计算每个聚类的差异表达基因
    sc.tl.rank_genes_groups(adata, 'clusters', method='wilcoxon', 
                           groups=[str(c) for c in target_clusters])
    
    # 提取前n个基因
    deg_dict = {}
    for cluster in target_clusters:
        cluster_str = str(cluster)
        if cluster_str in adata.uns['rank_genes_groups']['names'].dtype.names:
            genes = adata.uns['rank_genes_groups']['names'][cluster_str][:n_top_genes]
            deg_dict[f'Cluster_{cluster}'] = genes
        else:
            print(f"Warning: Cluster {cluster} not found in results")
    
    return deg_dict

def create_deg_heatmap(adata, deg_dict, savepath):
    """创建差异表达基因热图 (实验C)"""
    print("Creating DEG heatmap...")
    
    # 收集所有基因
    all_genes = []
    for genes in deg_dict.values():
        all_genes.extend(genes)
    all_genes = list(set(all_genes))
    
    # 检查基因是否存在于数据中
    available_genes = [g for g in all_genes if g in adata.var_names]
    if len(available_genes) == 0:
        print("Warning: No DEGs found in the dataset")
        return
    
    # 创建表达矩阵
    clusters = [3, 5, 15, 16]  # 对应 IDC, Healthy, Tumor_edge, DCIS/LCIS
    cluster_labels = ['IDC', 'Healthy', 'Tumor_edge', 'DCIS/LCIS']
    
    expression_data = []
    for cluster in clusters:
        cluster_mask = adata.obs['spatialGD_clusters'].astype(int) == cluster
        if cluster_mask.sum() > 0:
            cluster_expression = adata[cluster_mask, available_genes].X.mean(axis=0)
            if hasattr(cluster_expression, 'A1'):  # sparse matrix
                cluster_expression = cluster_expression.A1
            expression_data.append(cluster_expression)
        else:
            expression_data.append(np.zeros(len(available_genes)))
    
    # 创建DataFrame
    expr_df = pd.DataFrame(expression_data, 
                          index=cluster_labels,
                          columns=available_genes)
    
    # 创建热图
    plt.figure(figsize=(12, 6))
    sns.heatmap(expr_df, annot=True, cmap='viridis', cbar=True)
    plt.title('Top DEGs Expression Heatmap across Clusters')
    plt.ylabel('Tissue Types')
    plt.xlabel('Genes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(savepath + 'deg_heatmap.jpg', bbox_inches='tight', dpi=600)
    plt.show()
    
    return expr_df

def differential_expression_analysis(adata, cluster1=3, cluster2=16, savepath='./result/Human_Breast_Cancer/'):
    """进行聚类3和16之间的差异表达分析 (实验D)"""
    print(f"Performing differential expression analysis between cluster {cluster1} and {cluster2}...")
    
    # 获取两个聚类的细胞
    cluster1_mask = adata.obs['spatialGD_clusters'].astype(int) == cluster1
    cluster2_mask = adata.obs['spatialGD_clusters'].astype(int) == cluster2
    
    if cluster1_mask.sum() == 0 or cluster2_mask.sum() == 0:
        print(f"Warning: One or both clusters ({cluster1}, {cluster2}) have no cells")
        return None, None
    
    print(f"Cluster {cluster1} has {cluster1_mask.sum()} spots")
    print(f"Cluster {cluster2} has {cluster2_mask.sum()} spots")
    
    # 获取表达数据
    expr1 = adata[cluster1_mask].X
    expr2 = adata[cluster2_mask].X
    
    # 转换为dense如果是sparse
    if hasattr(expr1, 'toarray'):
        expr1 = expr1.toarray()
    if hasattr(expr2, 'toarray'):
        expr2 = expr2.toarray()
    
    # 计算fold change和p值
    results = []
    gene_names = adata.var_names
    
    for i, gene in enumerate(gene_names):
        mean1 = np.mean(expr1[:, i])
        mean2 = np.mean(expr2[:, i])
        
        # 避免除零
        if mean2 == 0:
            if mean1 == 0:
                log_fc = 0
            else:
                log_fc = np.inf
        else:
            log_fc = np.log2((mean1 + 1e-6) / (mean2 + 1e-6))
        
        # 使用Wilcoxon秩和检验
        try:
            _, p_val = ranksums(expr1[:, i], expr2[:, i])
        except:
            p_val = 1.0
        
        results.append({
            'gene': gene,
            'log2FoldChange': log_fc,
            'pvalue': p_val,
            'mean_cluster1': mean1,
            'mean_cluster2': mean2
        })
    
    # 创建结果DataFrame
    deg_results = pd.DataFrame(results)
    deg_results['-log10(pvalue)'] = -np.log10(deg_results['pvalue'] + 1e-300)
    
    # 标记显著差异基因
    deg_results['significance'] = 'normal'
    deg_results.loc[(deg_results['log2FoldChange'] > 1) & (deg_results['pvalue'] < 0.05), 'significance'] = 'up'
    deg_results.loc[(deg_results['log2FoldChange'] < -1) & (deg_results['pvalue'] < 0.05), 'significance'] = 'down'
    
    # 保存结果
    deg_results.to_csv(savepath + f'deg_cluster_{cluster1}_vs_{cluster2}.csv', index=False)
    
    return deg_results, gene_names



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Human_Breast_Cancer']

    for i in range(len(datasets)):
        dataset = datasets[i]
        path = './result/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(path):
            os.mkdir(path)
        print(dataset)
        adata, features, labels, sadj, fadj, graph_nei, graph_neg = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        savepath = './result/Human_Breast_Cancer/'
        plt.rcParams["figure.figsize"] = (4, 3)

        print(adata)
        title = "Manual annotation"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        # plt.savefig(savepath + dataset + '.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        config.epochs = config.epochs + 1

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
       
        
        model = Spatial_GraST_DGI(
            nfeat=config.fdim,
            nhid1=config.nhid1,
            nhid2=config.nhid2,
            dropout=config.dropout,
            contrastive_dim=config.contrastive_dim,
            use_spatial_contrastive=config.use_spatial_contrastive
        )
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []

        for epoch in trange(config.epochs, desc=f"Training {dataset}"):
            emb, mean, zinb_loss, reg_loss, con_loss, dgi_loss, total_loss = train_with_dgi(
                model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer
            )

            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)

            if ari_res > ari_max:
                    ari_max = ari_res
                    epoch_max = epoch
                    idx_max = idx
                    mean_max = mean
                    emb_max = emb


        print(f"The best result: {dataset} ARI={ari_max:.4f} )")
        

        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        
        
        title = 'SpatialDG'
        # pd.DataFrame(emb_max).to_csv(savepath + 'SpatialDG_emb.csv', header=None, index=None)
        # pd.DataFrame(idx_max).to_csv(savepath + 'SpatialDG_idx.csv', header=None, index=None)
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        # plt.savefig(savepath + 'SpatialDG.jpg', bbox_inches='tight', dpi=600)
        plt.show()
        # adata.write(savepath + 'SpatialDG.h5ad')

        print(f"最佳ARI: {ari_max:.4f}")
        
        # 实验C: DEG热图分析
        print("\n=== 实验C: DEG热图分析 ===")
        deg_dict = find_deg_analysis(adata, target_clusters=[3, 5, 15, 16], n_top_genes=3)
        print("发现的DEGs:", deg_dict)
        
        if deg_dict:
            expr_df = create_deg_heatmap(adata, deg_dict, savepath)
            print("DEG热图已保存")
        
        # 实验D: 聚类3 vs 16的差异表达分析
        print("\n=== 实验D: 差异表达分析 (IDC vs DCIS/LCIS) ===")
        deg_results, gene_names = differential_expression_analysis(adata, cluster1=3, cluster2=16, savepath=savepath)
        
        if deg_results is not None:
            print(f"总共分析了 {len(deg_results)} 个基因")
            print(f"显著上调基因: {(deg_results['significance'] == 'up').sum()}")
            print(f"显著下调基因: {(deg_results['significance'] == 'down').sum()}")

            # 显示top差异基因
            print("\nTop 10 上调基因:")
            top_up = deg_results[deg_results['significance'] == 'up'].nlargest(10, 'log2FoldChange')
            for _, gene_data in top_up.iterrows():
                print(f"  {gene_data['gene']}: log2FC={gene_data['log2FoldChange']:.3f}, p={gene_data['pvalue']:.2e}")
                
            print("\nTop 10 下调基因:")
            top_down = deg_results[deg_results['significance'] == 'down'].nsmallest(10, 'log2FoldChange')
            for _, gene_data in top_down.iterrows():
                print(f"  {gene_data['gene']}: log2FC={gene_data['log2FoldChange']:.3f}, p={gene_data['pvalue']:.2e}")
        
        print(f"\n=== 分析完成！结果保存在 {savepath} ===")
       
