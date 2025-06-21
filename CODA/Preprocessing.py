import scanpy as sc
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import bbknn


def preprocessing(ST1,ST2,integrate_method='combat',if_log='True'):

    gene1 = set(ST1.var_names)
    gene2 = set(ST2.var_names)
    common_gene = gene1.intersection(gene2)

    ST1 = ST1[:,list(common_gene)]
    ST2 = ST2[:,list(common_gene)]

    ST1.obs_names = ST1.obs_names.astype(str)
    ST2.obs_names = ST2.obs_names.astype(str)

    adata = ad.concat([ST1,ST2],label='batch',keys=['1','2'],index_unique='-')

    if if_log == 'True':
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        pass

    if integrate_method == 'combat':
        sc.pp.combat(adata, key='batch')

    elif integrate_method == 'bbknn':
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=30)
        bbknn.bbknn(adata, batch_key='batch')  
    elif integrate_method == None:
        pass

    else:
        print('error')

    batch_info = adata.obs['batch'].values
    adata_rep1 = adata[batch_info == '1']
    adata_rep2 = adata[batch_info == '2']
    
    return  adata_rep1,adata_rep2

def dimension_reduction(ST1,ST2,reduction_dim=3,method='umap'):

    adata = ad.concat([ST1,ST2],label='batch',keys=['1','2'],index_unique='-')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, n_components=reduction_dim)
    sc.pl.umap(adata, color='batch', title='UMAP colored by batch')
    umap_value = adata.obsm['X_umap']

    umap_min = umap_value.min(axis=0)
    umap_max = umap_value.max(axis=0)
    umap_normalized = (umap_value - umap_min) / (umap_max - umap_min)
    umap_rgb = np.round(umap_normalized * 255).astype(int)

    batch_info = adata.obs['batch'].values
    embedding_ST1 = umap_rgb[batch_info == '1']/255
    embedding_ST2 = umap_rgb[batch_info == '2']/255
    
    return embedding_ST1, embedding_ST2

def visualization(adata,embedding):
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        adata.obsm['spatial'][:, 0], 
        adata.obsm['spatial'][:, 1], 
        color=embedding[:,:3], 
        s=10 
    )
    plt.show()