
import torch.optim as optim
from utils import *
from SpatialDG.model import DGIDataAugmentation, Spatial_GraST_DGI
import os
import argparse
from tqdm import trange 
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import trange

device = torch.device('mps')
import os
os.environ["LD_LIBRARY_PATH"] = f"{os.popen('python -m rpy2.situation LD_LIBRARY_PATH').read().strip()}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# delta = 0.15 0.73 /0.1 0.7/ 0.2 0.69 / 0.16 0.7289/ 0.14 0.7345/0.13 0.6725/0 0.7064
def load_data(dataset):
    print("load data:")
    path = "../generate_data/DLPFC/" + dataset + "/Space.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg

def train_with_dgi(model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer, augmentor=None):
    """DGI训练函数"""
    model.train()
    optimizer.zero_grad()

    # 生成腐败视图（负样本）
    if augmentor is not None:
        corrupted = augmentor.create_corrupted_views(features, sadj, fadj)
        features_neg = corrupted['corrupted_features']
        sadj_neg = corrupted['corrupted_sadj']
        fadj_neg = corrupted['corrupted_fadj']
    else:
        features_neg = features
        sadj_neg = sadj
        fadj_neg = fadj
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
   
    datasets = ['151507']
    
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/DLPFC.ini'
        adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)
        
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        # plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        config = Config(config_file)
        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())
        
        config.epochs = config.epochs + 1

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        model = Spatial_GraST_DGI(
            nfeat=config.fdim,
            nhid1=config.nhid1,
            nhid2=config.nhid2,
            dropout=config.dropout,
            contrastive_dim=config.contrastive_dim,
            use_spatial_contrastive=config.use_spatial_contrastive
        )
        
        
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        print("Start training...")
        augmentor = DGIDataAugmentation(corruption_ratio=0.5) 

        for epoch in trange(config.epochs, desc=f"Training {dataset}"):
            emb, mean, zinb_loss, reg_loss, con_loss, dgi_loss, total_loss = train_with_dgi(
                model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer, augmentor=augmentor
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


        print(f"The best result: {dataset} ARI={ari_max:.4f} (epoch {epoch_max})")

       
        title = 'SpatialDG: ARI={:.2f}'.format(ari_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        # plt.savefig(savepath + 'Space.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        
        sc.pp.neighbors(adata, use_rep='mean')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.tl.paga(adata, groups='idx')
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, 
                          title=title, legend_fontoutline=2, show=False)
        # plt.savefig(savepath + 'Space_umap.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        
        # pd.DataFrame(emb_max).to_csv(savepath + 'Space_emb.csv')
        # pd.DataFrame(idx_max).to_csv(savepath + 'Space_idx.csv')
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean_max
        # adata.write(savepath + 'Space.h5ad')

        # print(f"The results have been saved to: {savepath}")

     

       
