import scanpy as sc
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
import bbknn
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
import scipy.sparse as sp

import warnings
warnings.filterwarnings('ignore')

def set_determinism(seed=0, num_threads=1):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def intersect_vars_in_order(adatas):
    common = adatas[0].var_names
    for a in adatas[1:]:
        common = common.intersection(a.var_names)
    return common

def ensure_obsnames_str(adatas):
    for a in adatas:
        a.obs_names = a.obs_names.astype(str)

def embedding_to_01(X, clip=True, eps=1e-12):
    X = np.asarray(X, dtype=float)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    denom = np.maximum(xmax - xmin, eps)
    Z = (X - xmin) / denom
    if clip:
        Z = np.clip(Z, 0, 1)
    return Z


def embedding_to_rgb01(X, clip=True, channels=None):
    X = np.asarray(X)
    n = X.shape[0]
    m = X.shape[1] if X.ndim == 2 else 1

    if channels is None:
        if m >= 3:
            X3 = X[:, :3]
        elif m == 2:
            X3 = np.c_[X, np.zeros((n, 1), dtype=X.dtype)]
        else:  # m == 1 或 0
            pad = np.zeros((n, max(0, 3 - m)), dtype=X.dtype)
            X3 = np.c_[X, pad]
    else:
        idx = tuple(channels) if isinstance(channels, (list, tuple, np.ndarray)) else (channels,)
        idx = idx[:3]
        X3 = np.zeros((n, 3), dtype=X.dtype)
        for c, k in enumerate(idx):
            k = int(k)
            if 0 <= k < m:
                X3[:, c] = X[:, k]

    return embedding_to_01(X3, clip=clip)


IntegrateMethod = Literal["combat", "bbknn", "none"]

@dataclass
class PreprocessConfig:
    integrate: IntegrateMethod = "combat"
    use_hvg: bool = True
    n_top_genes: int = 3000     
    normalize_target: float = 1e4
    log1p: bool = True
    scale_max_value: Optional[float] = 10.0  
    n_pcs: int = 30
    seed: int = 0
    num_threads: int = 1
    combat_densify_if_sparse: bool = True   

class STIntegrator:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def preprocess_and_integrate(
        self,
        adatas: List[ad.AnnData],
        batch_names: Optional[List[str]] = None,
        batch_key: str = "batch",
    ) -> ad.AnnData:
        assert len(adatas) >= 2, "Please provide at least two slices"
        set_determinism(self.cfg.seed, self.cfg.num_threads)

        for a in adatas:
            a.var_names_make_unique()
        ensure_obsnames_str(adatas)

        common = intersect_vars_in_order(adatas)
        adatas = [a[:, common].copy() for a in adatas]

        if batch_names is None:
            batch_names = [str(i) for i in range(len(adatas))]
        merged = ad.concat(
            adatas, label=batch_key, keys=batch_names, index_unique="-", merge="same"
        )

        if self.cfg.normalize_target:
            sc.pp.normalize_total(merged, target_sum=self.cfg.normalize_target)
        if self.cfg.log1p:
            sc.pp.log1p(merged)

        if self.cfg.use_hvg:
            sc.pp.highly_variable_genes(
                merged, n_top_genes=self.cfg.n_top_genes, flavor="seurat", batch_key=batch_key
            )
            merged = merged[:, merged.var["highly_variable"]].copy()

        if self.cfg.integrate == "combat":
            X = merged.X
            if sp.issparse(X) and self.cfg.combat_densify_if_sparse:
                merged.X = X.toarray()
            sc.pp.combat(merged, key=batch_key)

        elif self.cfg.integrate == "bbknn":
            if self.cfg.scale_max_value is not None:
                sc.pp.scale(merged, max_value=self.cfg.scale_max_value)
            sc.tl.pca(merged, n_comps=self.cfg.n_pcs, svd_solver="arpack", random_state=self.cfg.seed)
            import bbknn
            bbknn.bbknn(merged, batch_key=batch_key, neighbors_within_batch=3)  
        elif self.cfg.integrate == "none":
            pass
        else:
            raise ValueError(f"Unknown integrate method: {self.cfg.integrate}")

        return merged

    def embed(
        self,
        merged: ad.AnnData,
        method: Literal["umap", "pca"] = "umap",
        n_components: int = 3,
        batch_key: str = "batch",
    ):

        set_determinism(self.cfg.seed, self.cfg.num_threads)
    
        if method == "pca":
            sc.tl.pca(merged, n_comps=max(n_components, self.cfg.n_pcs),
                      svd_solver="arpack", random_state=self.cfg.seed)
            emb_raw = merged.obsm["X_pca"][:, :n_components]
        else:
            if "neighbors" not in merged.uns:
                sc.tl.pca(merged, n_comps=self.cfg.n_pcs,
                          svd_solver="arpack", random_state=self.cfg.seed)
                sc.pp.neighbors(merged, n_neighbors=15, n_pcs=self.cfg.n_pcs, random_state=self.cfg.seed)
            sc.tl.umap(merged, n_components=n_components, random_state=self.cfg.seed)
            emb_raw = merged.obsm["X_umap"]
    
        E = emb_raw.astype(np.float64)
        emin = E.min(axis=0)
        emax = E.max(axis=0)
        emb = ((E - emin) / np.maximum(emax - emin, 1e-12)).astype(np.float32)
    
        rgb01 = embedding_to_rgb01(emb)  
    
        batches = merged.obs[batch_key].astype(str).values
        uniq = pd.Index(sorted(pd.unique(batches)))
        split = {b: emb[batches == b] for b in uniq}
        split_rgb = {b: rgb01[batches == b] for b in uniq}
    
        return emb, rgb01, split, split_rgb