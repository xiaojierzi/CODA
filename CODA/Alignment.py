import scanpy as sc
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.transform import warp


'''
This part is about the global alignment.

'''
def normalization(coords):
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    coords_centered = coords - center
    max_abs_coords = np.abs(coords_centered).max(axis=0)
    spatial_normalized = coords_centered / max_abs_coords
    return spatial_normalized

def global_alignment(source_coords, target_coords, source_embedding, target_embedding):
    ''' 
    Parameters:    
    '''
    indices = nearest_neighbors(source_embedding,target_embedding)
    matched_dst = target_coords[indices]

    source_center = np.mean(source_coords, axis=0)
    target_center = np.mean(matched_dst, axis=0)

    source_centered = source_coords - source_center
    target_centered = matched_dst - target_center

    W = np.dot(source_centered.T, target_centered)
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(U, Vt)

    t = target_center - np.dot(R, source_center)
    return R, t

def nearest_neighbors(source_coords, target_coords):
    tree = KDTree(target_coords)
    dists, indices = tree.query(source_coords)
    return indices

'''
This part is about the local alignment.

'''

class LDDMM:

    def __init__(self, grid_shape: tuple[int, int] = (60, 60)) -> None:
        self.grid_shape = grid_shape

    def run_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        steps: int = 50,
        max_iter: int = 1000,
        sigma: float = 1.0,
        lr: float = 0.01,
    ):
        self.regulariser = Reguarizer(alpha, gamma)
        self.steps = steps
        H, W, C = source.shape[:3]

        v_seq = np.zeros((steps, H, W, 2))
        g_buf = np.copy(v_seq)
        energy_log: list[float] = []

        for it in range(max_iter):
            v_seq -= lr * g_buf
            if it % 10 == 9:
                v_seq = self._normalise_velocity(v_seq)

            inv_map = self._inverse_flow(v_seq)
            fwd_map = self._forward_flow(v_seq)

            src_evo = self._push_source(source, fwd_map)
            tgt_back = self._pull_target(target, inv_map)

            src_grad = self._image_grad(src_evo)
            jac_det = self._jac_det(inv_map)

            for t in range(steps):
                accum = np.zeros(self.grid_shape + (2,))
                diff = 2 / sigma**2 * jac_det[t][..., np.newaxis] * src_grad[t] * (
                    src_evo[t] - tgt_back[t]
                )[..., np.newaxis]
                for ch in range(C):
                    accum += self.regulariser.K(diff[:, :, ch, :])
                g_buf[t] = 2 * v_seq[t] - accum

            if np.linalg.norm(g_buf) < 1e-3:
                print("[Early Stop] ‖grad‖ < 1e‑3")
                break

            reg_e = sum(np.linalg.norm(self.regulariser.L(v_seq[t])) for t in range(steps))
            match_e = 1 / sigma**2 * np.sum((src_evo[-1] - target) ** 2)
            tot_e = reg_e + match_e
            energy_log.append(tot_e)
            print(f"Iter {it:03d} | Total {tot_e:.2f} | Reg {reg_e:.2f} | Match {match_e:.2f}")

        traj_len = sum(np.linalg.norm(self.regulariser.L(v_seq[t])) for t in range(steps))
        return src_evo[-1], v_seq, energy_log, traj_len, fwd_map, inv_map, src_evo, tgt_back

    def _normalise_velocity(self, v_seq: np.ndarray) -> np.ndarray:
        length = sum(np.linalg.norm(self.regulariser.L(v_seq[t])) for t in range(self.steps))
        for t in range(self.steps):
            denom = np.linalg.norm(self.regulariser.L(v_seq[t])) + 1e-8
            v_seq[t] = (length / self.steps) * v_seq[t] / denom
        return v_seq

    def _inverse_flow(self, v_seq: np.ndarray) -> np.ndarray:
        id_grid = coordinate_grid((v_seq.shape[1], v_seq.shape[2]))
        inv_flow = np.zeros_like(v_seq)
        inv_flow[-1] = id_grid
        for t in range(self.steps - 2, -1, -1):
            shift = self._inv_disp(v_seq[t], id_grid)
            inv_flow[t] = sample(inv_flow[t + 1], id_grid + shift)
        return inv_flow

    @staticmethod
    def _inv_disp(v_t: np.ndarray, x: np.ndarray) -> np.ndarray:
        alpha = np.zeros_like(v_t)
        for _ in range(5):
            alpha = sample(v_t, x + 0.5 * alpha)
        return alpha

    def _forward_flow(self, v_seq: np.ndarray) -> np.ndarray:
        id_grid = coordinate_grid((v_seq.shape[1], v_seq.shape[2]))
        fwd_flow = np.zeros_like(v_seq)
        fwd_flow[0] = id_grid
        for t in range(self.steps - 1):
            shift = self._fwd_disp(v_seq[t], id_grid)
            fwd_flow[t + 1] = sample(fwd_flow[t], id_grid - shift)
        return fwd_flow

    @staticmethod
    def _fwd_disp(v_t: np.ndarray, x: np.ndarray) -> np.ndarray:
        alpha = np.zeros_like(v_t)
        for _ in range(5):
            alpha = sample(v_t, x - 0.5 * alpha)
        return alpha

    def _push_source(self, img: np.ndarray, flow_seq: np.ndarray) -> np.ndarray:
        out = np.zeros((self.steps,) + img.shape)
        for t in range(self.steps):
            out[t] = sample(img, flow_seq[t])
        return out

    def _pull_target(self, img: np.ndarray, inv_seq: np.ndarray) -> np.ndarray:
        out = np.zeros((self.steps,) + img.shape)
        for t in reversed(range(self.steps)):
            out[t] = sample(img, inv_seq[t])
        return out

    def _image_grad(self, img_seq: np.ndarray) -> np.ndarray:
        grad_seq = np.zeros(img_seq.shape + (2,))
        for t in range(self.steps):
            grad_seq[t] = finite_difference(img_seq[t])
        return grad_seq

    def _jac_det(self, inv_seq: np.ndarray) -> np.ndarray:
        det = np.zeros(inv_seq.shape[:-1] + (1,))
        for t in range(self.steps):
            dx = finite_difference(inv_seq[t, :, :, 0])
            dy = finite_difference(inv_seq[t, :, :, 1])
            det[t] = dx[:, :, 0] * dy[:, :, 1] - dx[:, :, 1] * dy[:, :, 0]
        return det

    @staticmethod
    def check_topology_violation(det_seq: np.ndarray) -> bool:
        return det_seq.min() <= 0


class Reguarizer:

    def __init__(self, alpha: float, gamma: float):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._A_cache: np.ndarray | None = None
        

    def L(self, field: np.ndarray) -> np.ndarray:
        lap_kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        dxx = convolve(field[:, :, 0], lap_kernel)
        dyy = convolve(field[:, :, 1], lap_kernel)
        term = np.stack([dxx, dyy], axis=-1)
        return -self.alpha * term + self.gamma * field

    def K(self, g: np.ndarray) -> np.ndarray:
        if self._A_cache is None or self._A_cache.shape != g.shape[:-1]:
            self._A_cache = self._compute_A(g.shape)
        G = self._fft2(g)
        F = G / self._A_cache**2
        return self._ifft2(F)


    def _compute_A(self, shape: tuple[int, int, int]) -> np.ndarray:
        H, W, _ = shape
        kx = np.arange(H).reshape(-1, 1)
        ky = np.arange(W).reshape(1, -1)
        A = 2 * self.alpha * (
            (1 - np.cos(2 * np.pi * kx / H)) + (1 - np.cos(2 * np.pi * ky / W))
        ) + self.gamma
        return np.stack([A, A], axis=-1) 

    @staticmethod
    def _fft2(a: np.ndarray) -> np.ndarray:
        out = np.empty(a.shape, dtype=np.complex128)
        for c in range(a.shape[2]):
            out[:, :, c] = np.fft.fft2(a[:, :, c])
        return out

    @staticmethod
    def _ifft2(A: np.ndarray) -> np.ndarray:
        out = np.empty(A.shape, dtype=np.complex128)
        for c in range(A.shape[2]):
            out[:, :, c] = np.fft.ifft2(A[:, :, c])
        return np.real(out)

def coordinate_grid(shape: tuple[int, int]) -> np.ndarray:
    grid = np.mgrid[: shape[0], : shape[1]]
    return np.transpose(grid, (2, 1, 0))


def sample(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
    coords_t = np.transpose(coords, (2, 1, 0))  
    if img.ndim == 2:
        return warp(img, coords_t, mode="edge")
    else:
        channels = [warp(img[:, :, c], coords_t, mode="edge") for c in range(img.shape[2])]
        return np.stack(channels, axis=-1)


def finite_difference(arr: np.ndarray) -> np.ndarray:  
    kx = np.array([[1.0, 0.0, -1.0]])
    ky = kx.T
    mode = "reflect"

    def _grad2d(slice2d: np.ndarray) -> np.ndarray:
        gx = convolve(slice2d, kx, mode=mode)
        gy = convolve(slice2d, ky, mode=mode)
        return np.stack([gx, gy], axis=-1)

    if arr.ndim == 2:
        return _grad2d(arr)[..., np.newaxis] 
    elif arr.ndim == 3:
        H, W, C = arr.shape
        out = np.empty((H, W, C, 2), dtype=arr.dtype)
        for c in range(C):
            out[:, :, c, :] = _grad2d(arr[:, :, c])
        return out
    else:
        raise ValueError("Dimension error.")