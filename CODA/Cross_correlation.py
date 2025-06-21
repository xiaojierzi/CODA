import numpy as np
from scipy.spatial import distance_matrix
from typing import List, Tuple

class SpatialCrossCorrelation:
    def __init__(self, adata_1, adata_2,
                 coord_key_1='spatial', coord_key_2='spatial',
                 beta: float = 1.0):

        self.adata_1 = adata_1
        self.adata_2 = adata_2
        self.coords_1 = adata_1.obsm[coord_key_1]
        self.coords_2 = adata_2.obsm[coord_key_2]
        self.beta = beta

        self.genes = adata_1.var_names
        assert np.all(adata_1.var_names == adata_2.var_names),  "Gene sets must match."

        self.W = self._compute_weight_matrix()

    def _compute_weight_matrix(self) -> np.ndarray:
        dist = distance_matrix(self.coords_1, self.coords_2)
        return np.exp(-self.beta * dist ** 2)

    def compute_cross_correlation(self) -> List[Tuple[str, float]]:
        results = []
        W = self.W
        S_ij = np.sum(W)
        n = W.shape[0]

        for gene in self.genes:
            x = self.adata_1[:, gene].X.toarray().flatten() if hasattr(self.adata_1[:, gene].X, 'toarray') else self.adata_1[:, gene].X.flatten()
            y = self.adata_2[:, gene].X.toarray().flatten() if hasattr(self.adata_2[:, gene].X, 'toarray') else self.adata_2[:, gene].X.flatten()

            x_mean = x.mean()
            y_mean = y.mean()

            numerator = np.sum(W * np.outer(x - x_mean, y - y_mean))
            denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

            if denominator == 0:
                score = 0.0
            else:
                score = n / S_ij * numerator / denominator

            results.append((gene, score))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_top_genes(self, top_n: int = 20) -> List[Tuple[str, float]]:
        return self.compute_cross_correlation()[:top_n]