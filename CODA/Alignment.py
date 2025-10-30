import scanpy as sc
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.transform import warp
import skimage.transform
import scipy.sparse as sp


'''
This part is about the global alignment.
'''



def _center(X):
    mu = np.mean(X, axis=0)
    return X - mu, mu

def _enforce_proper_rotation(R, U, Vt):
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def estimate_rigid(src_xy, dst_xy):
    Xc, muX = _center(src_xy); Yc, muY = _center(dst_xy)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = _enforce_proper_rotation(Vt.T @ U.T, U, Vt)
    t = muY - R @ muX
    return R, t

def estimate_similarity(src_xy, dst_xy):
    Xc, muX = _center(src_xy); Yc, muY = _center(dst_xy)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = _enforce_proper_rotation(Vt.T @ U.T, U, Vt)
    s = np.sum(S) / np.sum(Xc**2)
    t = muY - s * (R @ muX)
    return s, R, t

def estimate_weak_affine(src_xy, dst_xy, A0=None, lam=1e-2):
    X, Y = src_xy, dst_xy
    n = X.shape[0]
    if A0 is None:
        s, R, _ = estimate_similarity(X, Y); A0 = s * R
    X_aug = np.hstack([X, np.ones((n, 1))])
    P = np.diag([1.0, 1.0, 0.0])
    Theta0 = np.zeros((3, 2)); Theta0[:2, :] = A0.T
    AtA = X_aug.T @ X_aug
    lhs = AtA + lam * (P.T @ P)
    rhs = X_aug.T @ Y + lam * (P.T @ P) @ Theta0
    Theta = np.linalg.solve(lhs, rhs)
    A = Theta[:2, :].T
    t = Theta[2, :]
    return A, t

def _apply_transform(X, mode, R=None, t=None, s=None, A=None):
    if mode == "G1":  return (X @ R.T) + t
    if mode == "G2":  return (s * (X @ R.T)) + t
    if mode == "G2+": return (X @ A.T) + t
    raise ValueError("mode must be 'G1'|'G2'|'G2+'")

def _labels_from_obs(adata, label_key, categories=None):
    if adata is None or label_key is None or label_key not in adata.obs:
        raise KeyError(f"obs does not contain column '{label_key}'")
    s = adata.obs[label_key].astype(object) 
    vals = s.fillna("__NA__").astype(str).to_numpy()

    if categories is None:
        cats = np.unique(vals)
    else:
        cats = np.array([str(c) for c in categories])

    cat2id = {c:i for i,c in enumerate(cats)}
    lab = np.array([cat2id.get(v, -1) for v in vals], dtype=int)
    lab = np.where(vals == "__NA__", -1, lab)
    return lab, cats

def _align_with_labels(src_xy, dst_xy, lab_s, lab_d, mode, trim_fraction, lam,
                       n_iter=5, min_pairs=50, min_shared_labels=2):
    common = np.intersect1d(np.unique(lab_s[lab_s >= 0]), np.unique(lab_d[lab_d >= 0]))
    ok = sum(((lab_s==l).sum() >= 3 and (lab_d==l).sum() >= 3) for l in common)
    if ok < min_shared_labels:
        raise ValueError(f"Insufficient common category (need ≥{min_shared_labels} and ≥3 points within the class), currently valid ={ok}")

    def _centroids_by_label(X, lab):
        C, L = [], []
        for l in np.unique(lab):
            if l < 0: continue
            m = (lab==l)
            if m.sum() >= 3:
                C.append(X[m].mean(axis=0)); L.append(l)
        return (np.vstack(C), np.array(L)) if C else (np.empty((0, X.shape[1])), np.array([], int))

    Cs, Ls = _centroids_by_label(src_xy, lab_s)
    Cd, Ld = _centroids_by_label(dst_xy, lab_d)
    labs = np.intersect1d(Ls, Ld)
    P = np.vstack([Cs[Ls==l][0] for l in labs])
    Q = np.vstack([Cd[Ld==l][0] for l in labs])

    if mode == "G1":
        R, t = estimate_rigid(P, Q); s, A = 1.0, None
    elif mode == "G2":
        s, R, t = estimate_similarity(P, Q); A = None
    else:
        s0, R0, _ = estimate_similarity(P, Q); A0 = s0 * R0
        A, t = estimate_weak_affine(P, Q, A0=A0, lam=lam); R, s = None, None

    pair_src = pair_dst = np.array([], dtype=int)
    for _ in range(n_iter):
        Xw = _apply_transform(src_xy, mode, R=R, t=t, s=s, A=A)
        trees, idxs = {}, {}
        for l in np.unique(lab_d):
            if l < 0: continue
            m = (lab_d==l)
            if m.sum() > 0:
                trees[l] = KDTree(dst_xy[m]); idxs[l] = np.nonzero(m)[0]

        ps, pd = [], []
        for i in range(src_xy.shape[0]):
            l = lab_s[i]
            if l < 0 or l not in trees: 
                continue
            j_local = trees[l].query(Xw[i])[1]
            ps.append(i); pd.append(idxs[l][j_local])

        if not ps: break
        ps = np.asarray(ps, int); pd = np.asarray(pd, int)
        Xp, Yp = src_xy[ps], dst_xy[pd]
        res = np.linalg.norm(Yp - Xw[ps], axis=1)

        if trim_fraction > 0:
            thr = np.quantile(res, 1 - trim_fraction)
            keep = res <= thr
            ps, pd, Xp, Yp = ps[keep], pd[keep], Xp[keep], Yp[keep]

        if Xp.shape[0] < 3: break

        if mode == "G1":
            R, t = estimate_rigid(Xp, Yp)
        elif mode == "G2":
            s, R, t = estimate_similarity(Xp, Yp)
        else:
            s0, R0, _ = estimate_similarity(Xp, Yp); A0 = s0 * R0
            A, t = estimate_weak_affine(Xp, Yp, A0=A0, lam=lam)

        pair_src, pair_dst = ps, pd

    aligned = _apply_transform(src_xy, mode, R=R, t=t, s=s, A=A)
    return {"mode": mode, "R": R, "t": t, "s": s, "A": A,
            "aligned": aligned, "pairs": (pair_src, pair_dst), "route": "labels"}


def align_points(
    src_xy, dst_xy,
    src_feat, dst_feat,
    mode="G1", trim_fraction=0.10, lam=1e-2,
    *,
    if_labeled=False,               
    label_key=None,                  
    src_adata=None, dst_adata=None,  
    n_iter=5, min_pairs=50, min_shared_labels=2
):

    if not if_labeled:
        A = src_feat.toarray() if sp.issparse(src_feat) else np.asarray(src_feat)
        B = dst_feat.toarray() if sp.issparse(dst_feat) else np.asarray(dst_feat)
        tree = KDTree(B)
        dists, idx = tree.query(A)
        if trim_fraction > 0:
            thr = np.quantile(dists, 1 - trim_fraction)
            mask = dists <= thr
        else:
            mask = np.ones_like(dists, dtype=bool)

        X = src_xy[mask]; Y = dst_xy[idx[mask]]
        assert X.shape[0] >= 3, "Too few valid pairs, check features or trim_fraction."

        if mode == "G1":
            R, t = estimate_rigid(X, Y); s, A_ = 1.0, None
            aligned = (src_xy @ R.T) + t
        elif mode == "G2":
            s, R, t = estimate_similarity(X, Y); A_ = None
            aligned = (s * (src_xy @ R.T)) + t
        else:
            s0, R0, _ = estimate_similarity(X, Y); A0 = s0 * R0
            A_, t = estimate_weak_affine(X, Y, A0=A0, lam=lam); R, s = None, None
            aligned = (src_xy @ A_.T) + t

        return {"mode": mode, "R": R, "t": t, "s": s, "A": A_,
                "indices": idx, "mask": mask, "aligned": aligned, "route": "features"}

    if src_adata is None or dst_adata is None or label_key is None:
        raise ValueError("If_labeled=True must provide src_adata, dst_adata, and label_key.")

    lab_s, cats_s = _labels_from_obs(src_adata, label_key, categories=None)
    lab_d, cats_d = _labels_from_obs(dst_adata, label_key, categories=None)
    cats_union = np.unique(np.concatenate([cats_s, cats_d]))
    lab_s, _ = _labels_from_obs(src_adata, label_key, categories=cats_union)
    lab_d, _ = _labels_from_obs(dst_adata, label_key, categories=cats_union)

    return _align_with_labels(
        src_xy, dst_xy, lab_s, lab_d,
        mode, trim_fraction, lam,
        n_iter=n_iter, min_pairs=min_pairs, min_shared_labels=min_shared_labels
    )




'''
This part is 3D reconstruction.
'''

def get_xy(adata, key='spatial'):
    return np.asarray(adata.obsm[key])

def get_feat(adata, feat_key='spatial'):
    return np.asarray(adata.obsm[feat_key])

def _can_use_labels(ad_prev, ad_curr, label_key, min_shared_labels=2, min_per_class=3):


    if (label_key is None or 
        (label_key not in ad_prev.obs) or 
        (label_key not in ad_curr.obs)):
        return False

    s1 = ad_prev.obs[label_key].astype(object).fillna("__NA__").astype(str)
    s2 = ad_curr.obs[label_key].astype(object).fillna("__NA__").astype(str)

    c1 = s1.value_counts()
    c2 = s2.value_counts()

    inter = c1.index.intersection(c2.index)
    inter = [k for k in inter if k != "__NA__"] 

    ok = sum((c1[k] >= min_per_class) and (c2[k] >= min_per_class) for k in inter)
    return ok >= min_shared_labels


def align_stack_chain(adatas, order=None, mode='G1', feat_key='feat', xy_keys=None,
                      trim_fraction=0.10, lam=1e-2,
                      if_labeled=False, label_key=None, min_shared_labels=2, verbose=False):

    n = len(adatas)
    if order is None: order = list(range(n))
    if xy_keys is None: xy_keys = ['spatial'] * n

    params = {}

    start_id = order[0]
    start = adatas[start_id]
    start.obsm['spatial_aligned'] = get_xy(start, xy_keys[start_id]).copy()
    params[start_id] = {"mode":"ID","R":np.eye(2),"t":np.zeros(2),"s":1.0,"A":np.eye(2)}

    for i in range(1, len(order)):
        k_prev, k_curr = order[i-1], order[i]
        prev, curr = adatas[k_prev], adatas[k_curr]

        XY_prev = prev.obsm['spatial_aligned']
        XY_curr = get_xy(curr, xy_keys[k_curr])

        use_labels_now = if_labeled and _can_use_labels(prev, curr, label_key, min_shared_labels)

        if use_labels_now:
            if verbose:
                print(f"[{k_prev}->{k_curr}] label-based alignment on '{label_key}'")
            res = align_points(
                XY_curr, XY_prev,
                src_feat=None, dst_feat=None,
                mode=mode, trim_fraction=trim_fraction, lam=lam,
                if_labeled=True, label_key=label_key,
                src_adata=curr, dst_adata=prev
            )
        else:
            if verbose and if_labeled:
                print(f"[{k_prev}->{k_curr}] labels unavailable/insufficient -> fallback to feature-based alignment")
            F_prev  = get_feat(prev, feat_key)
            F_curr  = get_feat(curr, feat_key)
            res = align_points(
                XY_curr, XY_prev,
                F_curr, F_prev,
                mode=mode, trim_fraction=trim_fraction, lam=lam
            )

        curr.obsm['spatial_aligned'] = res['aligned']
        params[k_curr] = res

    return adatas, params, start_id

def _stack_to_df(adatas, label_key='Layer_Guess', section_names=None):
    frames = []
    for i, ad in enumerate(adatas):
        if 'spatial_3d' not in ad.obsm:
            raise ValueError(f"The {i}th AnnData is missing obsm['spatial_3d']")
        xyz = ad.obsm['spatial_3d']
        df = pd.DataFrame(xyz, columns=['x','y','z'])
        if label_key not in ad.obs:
            raise ValueError(f"There is no '{label_key}' column in obs")
        df['label'] = ad.obs[label_key].astype(str).values

        sec = section_names[i] if section_names else (ad.obs.get('section_id', pd.Series([f's{i+1}'], index=ad.obs_names)).iloc[0])
        df['section'] = sec
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def plot_3d_alignment_by_label(adatas, label_key='Layer_Guess', section_names=None,
                               subsample=80000, point_size=2, elev=20, azim=60):
    df = _stack_to_df(adatas, label_key, section_names)
    if len(df) > subsample:
        df = df.sample(subsample, random_state=0)

    cats  = pd.Categorical(df['label'])
    codes = cats.codes.astype(float)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(df['x'], df['y'], df['z'],
                    c=codes, s=point_size, alpha=0.9, depthshade=False)

    ax.view_init(elev=elev, azim=azim)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.grid(False)
    try:
        ax.xaxis.set_pane_color((1,1,1,0))
        ax.yaxis.set_pane_color((1,1,1,0))
        ax.zaxis.set_pane_color((1,1,1,0))
        ax.xaxis.line.set_color((1,1,1,0))
        ax.yaxis.line.set_color((1,1,1,0))
        ax.zaxis.line.set_color((1,1,1,0))
    except Exception:
        pass
    try:
        ax.set_axis_off()
    except Exception:
        pass

    xr = float(df['x'].max() - df['x'].min())
    yr = float(df['y'].max() - df['y'].min())
    zr = float(df['z'].max() - df['z'].min())
    ax.set_box_aspect((xr, yr, zr))

    fig.patch.set_alpha(0.0)
    plt.show()


'''
This part is about the local alignment.

'''


class _BiharmonicOp:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self._A = None
        self._A2 = None  

    def apply_L(self, f):
        w = np.array([[0.0, 1.0, 0.0],
                      [1.0, -4.0, 1.0],
                      [0.0, 1.0, 0.0]])
        dxx = convolve(f[:, :, 0], w)
        dyy = convolve(f[:, :, 1], w)
        dff = np.stack([dxx, dyy], axis=-1)
        return -self.alpha * dff + self.gamma * f

    def apply_K(self, g):
        if (self._A is None) or (self._A.shape != g.shape):
            self._compute_A(g.shape)

        G = self._fft2(g)
        F = G / self._A2
        return self._ifft2(F)

    def _compute_A(self, shape):
        H, W, d = shape
        kh = np.arange(H, dtype=np.float64)[:, None]
        kw = np.arange(W, dtype=np.float64)[None, :]
        base = (1 - np.cos(2 * np.pi * kh / H)) + (1 - np.cos(2 * np.pi * kw / W))
        A = 2 * self.alpha * base + self.gamma
        A = np.stack([A, A], axis=-1)
        self._A = A
        self._A2 = A * A
        return A

    @staticmethod
    def _fft2(a):
        return np.fft.fft2(a, axes=(0, 1))

    @staticmethod
    def _ifft2(A):
        return np.real(np.fft.ifft2(A, axes=(0, 1)))

def _resample(array, coordinates):
    coords = np.transpose(coordinates, axes=[2, 1, 0])  # identical permutation

    if array.ndim == 2:
        return skimage.transform.warp(array, coords, mode='edge')
    elif array.ndim == 3:
        C = array.shape[-1]
        out = []
        for c in range(C):
            out.append(skimage.transform.warp(array[:, :, c], coords, mode='edge'))
        return np.stack(out, axis=-1)


def _grid(shape):
    g = np.mgrid[:shape[0], :shape[1]]
    return np.transpose(g, axes=[2, 1, 0])


def _finite_diff(a):
    wx = np.array([[1.0, 0.0, -1.0]])
    wy = wx.T
    mode = 'reflect'

    def _grad(z):
        gx = convolve(z, wx, mode=mode)
        gy = convolve(z, wy, mode=mode)
        return np.stack([gx, gy], axis=-1)

    if a.ndim == 2:
        return _grad(a)[..., np.newaxis]  # (H, W, 2, 1)
    elif a.ndim == 3:
        H, W, C = a.shape
        G = np.empty((H, W, C, 2))
        for c in range(C):
            G[:, :, c, :] = _grad(a[:, :, c])
        return G

class LDDMM(object):
    def register(self, I0, I1, T=32, K=200, sigma=1, alpha=1, gamma=1, epsilon=0.01):
        self.T = T
        self.K = K
        self.H, self.W, self.c = I0.shape[:3]
        self.regularizer = _BiharmonicOp(alpha, gamma)
        energies = []

        v = np.zeros((T, self.H, self.W, 2))
        dv = np.copy(v)

        for k in range(K):
            v -= epsilon * dv
            if k % 10 == 9:
                v = self.reparameterize(v)

            Phi1 = self.integrate_backward_flow(v)
            Phi0 = self.integrate_forward_flow(v)

            J0 = self.push_forward(I0, Phi0)
            J1 = self.pull_back(I1, Phi1)

            dJ0 = self.image_grad(J0)
            detPhi1 = self.jacobian_derterminant(Phi1)

            for t in range(0, self.T):
                accum = np.zeros((self.H, self.W, 2))
                tmp = (2 / sigma**2) * detPhi1[t][:, :, :, np.newaxis] * dJ0[t] * (J0[t] - J1[t])[:, :, :, np.newaxis]
                for c in range(self.c):
                    sl = tmp[:, :, c, :]
                    accum += self.regularizer.apply_K(sl)
                dv[t] = 2 * v[t] - accum

            dv_norm = np.linalg.norm(dv)
            if dv_norm < 0.001:
                print(dv_norm)
                print("gradient norm below threshold. stopping")
                break

            E_reg = np.sum([np.linalg.norm(self.regularizer.apply_L(v[t])) for t in range(T)])
            E_int = (1 / sigma**2) * np.sum((J0[-1] - I1) ** 2)
            E = E_reg + E_int
            energies.append(E)

            print(
                "iteration {:3d}, energy {:4.2f}, thereof {:4.2f} regularization and {:4.2f} intensity difference".format(
                    k, E, E_reg, E_int
                )
            )

        v_hat = v
        length = np.sum([np.linalg.norm(self.regularizer.apply_L(v_hat[t])) for t in range(T)])
        return J0[-1], v_hat, energies, length, Phi0, Phi1, J0, J1

    def reparameterize(self, v):
        length = np.sum([np.linalg.norm(self.regularizer.apply_L(v[t])) for t in range(self.T)])
        for t in range(self.T):
            v[t] = length / self.T * v[t] / np.linalg.norm(self.regularizer.apply_L(v[t]))
        return v

    def integrate_backward_flow(self, v):
        x = _grid((self.H, self.W))
        Phi1 = np.zeros((self.T, self.H, self.W, 2))
        Phi1[self.T - 1] = x
        for t in range(self.T - 2, -1, -1):
            alpha = self.backwards_alpha(v[t], x)
            Phi1[t] = _resample(Phi1[t + 1], x + alpha)
        return Phi1

    def backwards_alpha(self, v_t, x):
        alpha = np.zeros(v_t.shape)
        for _ in range(5):
            alpha = _resample(v_t, x + 0.5 * alpha)
        return alpha

    def integrate_forward_flow(self, v):
        x = _grid((self.H, self.W))
        Phi0 = np.zeros((self.T, self.H, self.W, 2))
        Phi0[0] = x
        for t in range(0, self.T - 1):
            alpha = self.forward_alpha(v[t], x)
            Phi0[t + 1] = _resample(Phi0[t], x - alpha)
        return Phi0

    def forward_alpha(self, v_t, x):
        alpha = np.zeros(v_t.shape)
        for _ in range(5):
            alpha = _resample(v_t, x - 0.5 * alpha)
        return alpha

    def push_forward(self, I0, Phi0):
        J0 = np.zeros((self.T,) + I0.shape)
        for t in range(0, self.T):
            J0[t] = _resample(I0, Phi0[t])
        return J0

    def pull_back(self, I1, Phi1):
        J1 = np.zeros((self.T,) + I1.shape)
        for t in range(self.T - 1, -1, -1):
            J1[t] = _resample(I1, Phi1[t])
        return J1

    def image_grad(self, J0):
        dJ0 = np.zeros(J0.shape + (2,))
        for t in range(self.T):
            dJ0[t] = _finite_diff(J0[t])
        return dJ0

    def jacobian_derterminant(self, Phi1):
        detPhi1 = np.zeros((self.T, self.H, self.W, 1))
        for t in range(self.T):
            dx = _finite_diff(Phi1[t, :, :, 0])
            dy = _finite_diff(Phi1[t, :, :, 1])
            detPhi1[t] = dx[:, :, 0] * dy[:, :, 1] - dx[:, :, 1] * dy[:, :, 0]
        return detPhi1

    def is_injectivity_violated(self, detPhi1):
        return detPhi1.min() < 0