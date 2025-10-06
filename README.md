# SpatialDG: A Novel Spatial Domain Identification Method For Spatially-resolved Transcriptomics Data Based On Dual-Graph Neural Network
<img width="1507" height="844" alt="figure1" src="https://github.com/user-attachments/assets/15707a05-5401-4f87-bb3d-255d38530a47" />
SpatialDG is a dual-graph self-supervised contrastive learning framework for ST. SpatialDG combines graph neural networks with self-supervised contrastive learning to learn informative and discriminative spot representations by maximizing the agreement between local (node) and global (graph) embeddings, while leveraging spatial adjacency to enhance representation learning.

## Requirements:
SpatialDG is implemented in the pytorch framework. The following packages are required to be able to run everything in this repository (included are the versions we used):

Python                            3.12.2

anndata                           0.11.3

matplotlib                        3.9.2

matplotlib-inline                 0.1.6

numpy                             1.26.4

numpydoc                          1.7.0

pandas                            2.2.2

scanpy                            1.11.0

scikit-learn                      1.5.1

scipy                             1.13.1

seaborn                           0.13.2

torch                             2.5.1

torchvision                       0.20.1

tqdm                              4.66.5

## Data

All the datasets used in this paper are open access and can be downloaded from the following websites: 

(1) The LIBD human dorsolateral prefrontal cortex (DLPFC) dataset (http://spatial.libd.org/spatialLIBD)

(2) 10x Visium spatial transcriptomics dataset of human breast cancer (https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1)

(3) Mouse embryo Stereo-seq data (https://db.cngb.org/stomics/mosta/)

## SpatialDG — How to run on HBC, DLPFC, Embryo

This guide shows how to:
- HBC: generate data, then test
- DLPFC: generate data, then test
- Embryo: run directly (no generate step)

### 0) Setup
- macOS Terminal
- Python 3.9+
- Create and activate env (example):
```bash
conda create -n spatialdg python=3.9 -y
conda activate spatialdg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scanpy anndata numpy scipy pandas scikit-learn matplotlib seaborn tqdm
# optional (for GATv2):
# pip install torch-geometric
```

### 1) Data layout (expected by scripts)
Project root:
```
/Users/apple/Desktop/SpatialGD
```

Place raw data as:
- HBC:
  - /Users/apple/Desktop/data/Human_Breast_Cancer/
    - filtered_feature_bc_matrix.h5
    - metadata.tsv
    - spatial/ (Visium images)
- DLPFC:
  - /Users/apple/Desktop/data/DLPFC/<slice_id>/
    - filtered_feature_bc_matrix.h5
    - metadata.tsv (has column layer_guess_reordered)
    - spatial/
  - Example slice_id: 151507, 151676, ...
- Embryo:
  - /Users/apple/Desktop/data/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad

Note: scripts read from ../data relative to SpatialDG folder.

### 2) Human Breast Cancer (HBC)
Step A — Generate graphs:
```bash
cd /Users/apple/Desktop/SpatialGD
python SpatialDG/HBC_generate_data.py
```
- Reads: ../data/Human_Breast_Cancer/
- Writes: ../generate_data/Human_Breast_Cancer/Space.h5ad

Step B — Test (training + visualization):
```bash
python SpatialDG/HBC_test.py
```
- Reads: ../generate_data/Human_Breast_Cancer/Space.h5ad
- Results: figures printed; optional saves in ./result/Human_Breast_Cancer/

### 3) DLPFC
Step A — Generate graphs:
```bash
cd /Users/apple/Desktop/SpatialGD
# Edit the target slice(s) in SpatialDG/DLPFC_generate_data.py (datasets list), e.g. ['151676']
python SpatialDG/DLPFC_generate_data.py
```
- Reads: ../data/DLPFC/<slice_id>/
- Writes: ../generate_data/DLPFC/<slice_id>/Space.h5ad

Step B — Test (training + ARI evaluation):
```bash
# Edit the slice in SpatialDG/dlpfctest.py (datasets list), e.g. ['151507']
python SpatialDG/dlpfctest.py
```
- Reads: ../generate_data/DLPFC/<slice_id>/Space.h5ad
- Results: figures printed; optional saves in ./result/DLPFC/<slice_id>/

Tips:
- DLPFC_generate_data.py uses the config file at ./config/DLPFC.ini for k, radius, HVGs.
- dlpfctest.py computes ARI vs manual annotation and shows spatial/UMAP/PAGA plots.

### 4) Mouse Embryo (Embryo)
Embryo does not need a generate step; it builds graphs on the fly.

Run:
```bash
cd /Users/apple/Desktop/SpatialGD
python SpatialDG/embryo.py
```
- Reads: ../data/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad
- Produces clustering and per-domain plots; optional saves in ./result/Mouse_Embryo/

### 5) Notes
- If R/mclust is unavailable, embryo.py will fallback to KMeans (try/except).
- To disable saving, keep .write/.to_csv/.savefig commented (already mostly commented in scripts).
- For MPS (Apple Silicon), scripts set device('mps') where applicable.
- Colors: ensure plot_color length >= number of domains.

### 6) Change dataset slices
- DLPFC:
  - SpatialDG/DLPFC_generate_data.py → datasets list (default ['151676'])
  - SpatialDG/dlpfctest.py → datasets list (default ['151507'])
- HBC:
  - Fixed dataset name 'Human_Breast_Cancer' in both HBC_generate_data.py and HBC_test.py

### 7) Troubleshooting
- Scanpy plotting returns ax=None when show=True → set show=False to get the axis.
- GATv2 with sparse adj → convert adj.to_dense() inside the layer before attention ops.
- Marker genes not shown → check names in adata.var_names; set use_raw/layer if needed.
