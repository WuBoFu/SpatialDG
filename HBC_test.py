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
        # pd.DataFrame(idx_max).to_csv(savepath + 'SpatialDG_idx.csv', header=None, index=None
        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        # plt.savefig(savepath + 'SpatialDG.jpg', bbox_inches='tight', dpi=600)
        plt.show()
        # adata.write(savepath + 'SpatialDG.h5ad')

        print(f"最佳ARI: {ari_max:.4f}")

       
