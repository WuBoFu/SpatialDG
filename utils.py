import os
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

# 可选：mclust 聚类
try:
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    _R_AVAILABLE = True
except Exception:
    _R_AVAILABLE = False

# -----------------------------
# 相似度/正则与一致性损失
# -----------------------------
def _nan2zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

def cosine_similarity(emb: torch.Tensor) -> torch.Tensor:
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((-1, 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat

def regularization_loss(emb: torch.Tensor, graph_nei: torch.Tensor, graph_neg: torch.Tensor) -> torch.Tensor:
    sim = torch.sigmoid(cosine_similarity(emb))
    neigh_loss = (graph_nei * torch.log(sim + 1e-12)).mean()
    neg_loss = (graph_neg * torch.log(1 - sim + 1e-12)).mean()
    return -0.5 * (neigh_loss + neg_loss)

def consistency_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
    emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = emb1 @ emb1.T
    cov2 = emb2 @ emb2.T
    return torch.mean((cov1 - cov2) ** 2)

# -----------------------------
# 图构建
# -----------------------------
def features_construct_graph(features: np.ndarray, k: int = 15, pca: int | None = None,
                             mode: str = "connectivity", metric: str = "cosine"):
    """基于表达特征构建 KNN 图（可选 PCA 降维）"""
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True).toarray()
    # 去自环
    i, j = np.diag_indices_from(A)
    A[i, j] = 0
    # 无向化
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj

def spatial_construct_graph_radius(positions: np.ndarray, radius: float = 150.0):
    """基于空间坐标的半径邻接图"""
    nbrs = NearestNeighbors(radius=radius).fit(positions)
    distances, indices = nbrs.radius_neighbors(positions, return_distance=True)

    n = positions.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for it in range(n):
        A[it, indices[it]] = 1.0
    # 去自环
    np.fill_diagonal(A, 0.0)

    # 无向化
    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    # 正负样本掩码（张量）
    graph_nei = torch.tensor(A, dtype=torch.float32)
    graph_neg = torch.ones_like(graph_nei) - graph_nei
    return sadj, graph_nei, graph_neg

# -----------------------------
# 稀疏工具
# -----------------------------
def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """scipy sparse -> torch sparse"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_sparse_matrix(mx: sp.spmatrix) -> sp.spmatrix:
    """按行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def dopca(data: np.ndarray, dim: int = 50) -> np.ndarray:
    return PCA(n_components=dim).fit_transform(data)

# -----------------------------
# ZINB / NB 损失
# -----------------------------
class NB(object):
    def __init__(self, theta=None, scale_factor: float = 1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, mean: bool = True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6, device=y_true.device, dtype=y_true.dtype))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + self.eps)) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps))
        )
        final = t1 + t2
        final = _nan2inf(final)
        return torch.mean(final) if mean else final

class ZINB(NB):
    def __init__(self, pi: torch.Tensor, ridge_lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, mean: bool = True):
        scale = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6, device=y_true.device, dtype=y_true.dtype))

        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)

        ridge = self.ridge_lambda * torch.square(self.pi)
        result = result + ridge
        result = _nan2inf(result)
        return torch.mean(result) if mean else result

# -----------------------------
# 聚类（mclust，可选）
# -----------------------------
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    """调用 R 的 mclust 对 adata.obsm[used_obsm] 聚类，写入 adata.obs['mclust']"""
    if not _R_AVAILABLE:
        raise RuntimeError("rpy2/mclust not available.")
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri as numpy2ri
    numpy2ri.activate()
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']

    emb = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    res = rmclust(numpy2ri.numpy2rpy(emb), num_cluster, modelNames)
    if len(res) < 2 or res[-2] is None:
        raise ValueError("Mclust failed, check input and cluster number.")
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res.astype(int).astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', method='mclust'):
    """优先用 mclust；失败时由上层脚本回退到 KMeans。"""
    # 降维以稳定 mclust
    emb = np.asarray(adata.obsm[key])
    adata.obsm['emb_pca'] = PCA(n_components=min(20, emb.shape[1]), random_state=42).fit_transform(emb)
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
