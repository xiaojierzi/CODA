import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as path_effects
from PIL import Image
from IPython.display import Image as IPyImage, display
from pathlib import Path
import torch,cv2
from shapely.geometry import Polygon as SPolygon
from matplotlib.patches import Rectangle, Polygon
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from typing import Optional


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

def embedding_to_01(X, clip=True, eps=1e-12):
    X = np.asarray(X, dtype=float)
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    denom = np.maximum(xmax - xmin, eps)
    Z = (X - xmin) / denom
    if clip:
        Z = np.clip(Z, 0, 1)
    return Z

def _batch_arrays(merged, batch_key="batch"):
    batches = merged.obs[batch_key].astype(str).values
    uniq = np.unique(batches)
    return batches, uniq

def plot_batches_in_embedding_2d(
    merged, emb, batch_key="batch",
    point_size=6, alpha=0.7, title=None,
    dims=(0, 1), show_legend=True
):
    assert len(dims) == 2, "dims must contain two dimension indices"
    D = emb.shape[1]
    i, j = int(dims[0]), int(dims[1])
    assert 0 <= i < D and 0 <= j < D, f"dims out of bounds: embedding dimension is {D}, but dims={dims}"

    x, y = emb[:, i], emb[:, j]
    batches, uniq = _batch_arrays(merged, batch_key=batch_key)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    for b in uniq:
        idx = (batches == b)
        ax.scatter(x[idx], y[idx], s=point_size, alpha=alpha, label=str(b))
    ax.set_xlabel(f"dim {i+1}"); ax.set_ylabel(f"dim {j+1}")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(title=batch_key, markerscale=3)
    plt.show()


def plot_batches_in_embedding_3d(
    merged, emb, batch_key="batch",
    point_size=6, alpha=0.7, title=None,
    dims=(0, 1, 2), show_legend=True
):
 
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  
    assert len(dims) == 3, "dims must contain three dimension indices"
    D = emb.shape[1]
    i, j, k = map(int, dims)
    assert 0 <= i < D and 0 <= j < D and 0 <= k < D, f"dims out of bounds: embedding dimension is {D}, but dims={dims}"

    x, y, z = emb[:, i], emb[:, j], emb[:, k]
    batches, uniq = _batch_arrays(merged, batch_key=batch_key)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for b in uniq:
        idx = (batches == b)
        ax.scatter(x[idx], y[idx], z[idx], s=point_size, alpha=alpha, label=str(b))
    ax.set_xlabel(f"dim {i+1}"); ax.set_ylabel(f"dim {j+1}"); ax.set_zlabel(f"dim {k+1}")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(title=batch_key, loc="best")
    plt.show()

def plot_slice_rgb(
    merged,
    rgb,
    batch,
    xy_key="spatial",
    batch_key="batch",
    s=1,
    alpha=1.0,
    invert_y=False,
    title=None,
    channels=None,           
    remap_if_not_3d=True,    
    quantize=False           
):
    b = str(batch).strip()


    if batch_key not in merged.obs:
        raise ValueError(f"batch_key='{batch_key}' not found in obs. Available columns: {list(merged.obs.columns)}")
    batches = merged.obs[batch_key].astype(str).str.strip().values
    mask = (batches == b)
    if mask.sum() == 0:
        uniq = np.unique(batches).tolist()
        raise ValueError(f"batch '{b}' has no entries. Available batches: {uniq}")

    if xy_key not in merged.obsm:
        raise ValueError(f"xy_key='{xy_key}' not found in obsm. Available keys: {list(merged.obsm.keys())}")


    if getattr(rgb, "shape", None) is None or rgb.ndim != 2:
        raise ValueError("`rgb` needs to be a 2D array: (N,*) or (n_batch,*).")

    if rgb.shape[0] == merged.n_obs:
        C_raw = rgb[mask]
    elif rgb.shape[0] == int(mask.sum()):
        C_raw = rgb
    else:
        raise ValueError(f"`rgb` number of rows ({rgb.shape[0]}) should be equal to merged.n_obs({merged.n_obs}) or the batch size ({int(mask.sum())}).")

    if C_raw.shape[1] == 3:
        C = C_raw.astype(float)
        if C.max() > 1.0:
            C /= 255.0
    else:
        if not remap_if_not_3d:
            raise ValueError("The passed in color is not a 3-channel color; set remap_if_not_3d=True or map to RGB yourself first.")
        C = embedding_to_rgb01(C_raw, channels=channels)  

    if quantize:
        C = np.round(C * 255) / 255.0

    XY = merged.obsm[xy_key][mask]
    plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.scatter(XY[:, 0], XY[:, 1], c=C, s=s, alpha=alpha)
    if invert_y:
        ax.invert_yaxis()
    ax.set_title(title or f"{xy_key} RGB — {batch_key}={b}")
    plt.show()


def generate_gif_from_sequence(img_seq: np.ndarray,
                                gif_path: str = "transformation.gif",
                                duration: int = 100,
                                display_inline: bool = True):

    if img_seq.ndim == 3:
        img_seq = img_seq[..., np.newaxis]

    if img_seq.max() <= 1.0:
        img_seq = (img_seq * 255).astype(np.uint8)
    else:
        img_seq = img_seq.astype(np.uint8)

    frames = []
    for frame in img_seq:
        fig, ax = plt.subplots()
        if frame.shape[-1] == 1:
            ax.imshow(frame[..., 0], cmap='gray')
        else:
            ax.imshow(frame)
        ax.axis('off')
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(buf))
        plt.close(fig)

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    frames[0].save(gif_path, format='GIF', append_images=frames[1:],
                   save_all=True, duration=duration, loop=0)

    if display_inline:
        display(IPyImage(filename=gif_path))


def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)

def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  
    elif image.ndim == 2:
        image = image[None]  
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def rbd(data: dict) -> dict:
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }



def cm_RdGn(x):
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_BlRdGn(x_):
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 0.1, 1, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)
    return out


def cm_prune(x_):
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    max_i = max(x_)
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)
    return cm_BlRdGn(norm_x)


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    imgs = [
        img.permute(1, 2, 0).cpu().numpy()
        if (isinstance(img, torch.Tensor) and img.dim() == 3)
        else img
        for img in imgs
    ]

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values(): 
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
    return ax 


def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0):
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def extract_and_match_keypoints(
    image0, image1, extractor, matcher, device="cuda"
):
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0 = kpts0[matches[..., 0]]
    m_kpts1 = kpts1[matches[..., 1]]

    axes = plot_images([image0, image1])
    plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2, axes=axes)

    return m_kpts0, m_kpts1, matches01


def plot_match_regions(
    image0, image1,
    kpts0: torch.Tensor, kpts1: torch.Tensor,
    matches, prune0, prune1,
    mode: str = "bbox",
    lw: float = 2.0
):
    kpts0_np = kpts0.detach().cpu().numpy()
    kpts1_np = kpts1.detach().cpu().numpy()

    axes = plot_images([image0, image1])
    plot_matches(kpts0, kpts1, color="lime", lw=0.2, axes=axes)  

    bbox0 = bbox1 = None  
    if mode == "bbox":
        x0_min, x0_max = kpts0_np[:, 0].min(), kpts0_np[:, 0].max()
        y0_min, y0_max = kpts0_np[:, 1].min(), kpts0_np[:, 1].max()
        x1_min, x1_max = kpts1_np[:, 0].min(), kpts1_np[:, 0].max()
        y1_min, y1_max = kpts1_np[:, 1].min(), kpts1_np[:, 1].max()

        plot_rect(axes[0], x0_min, x0_max, y0_min, y0_max, color="red",  lw=lw)
        plot_rect(axes[1], x1_min, x1_max, y1_min, y1_max, color="blue", lw=lw)

        bbox0 = (x0_min, y0_min, x0_max, y0_max)
        bbox1 = (x1_min, y1_min, x1_max, y1_max)

    elif mode == "convex":
        def plot_hull(ax, points, color):
            if len(points) >= 3:
                hull = ConvexHull(points)
                poly = SPolygon(points[hull.vertices])
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color=color, linewidth=lw)
        plot_hull(axes[0], kpts0_np, color="red")
        plot_hull(axes[1], kpts1_np, color="blue")
    else:
        raise ValueError(f"Unsupported mode: {mode}, use 'bbox' or 'convex'.")

    kpc0 = cm_prune(prune0)
    kpc1 = cm_prune(prune1)

    return axes, bbox0, bbox1   

def plot_rect(ax, x_min, x_max, y_min, y_max, color="red", lw=2):
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=lw, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def extract_cells_from_polygon(polygon, mapping):
    cell_indices = set()

    height = len(mapping)
    width = len(mapping[0])

    minx, miny, maxx, maxy = polygon.bounds
    minx, maxx = int(np.floor(minx)), int(np.ceil(maxx))
    miny, maxy = int(np.floor(miny)), int(np.ceil(maxy))

    for y in range(miny, maxy + 1):
        if y < 0 or y >= height:
            continue
        for x in range(minx, maxx + 1):
            if x < 0 or x >= width:
                continue
            if polygon.contains(Point(x, y)):
                cell_indices.update(mapping[y][x]) 

    return cell_indices



def extract_cells_from_matrix_window(mapping, r0, r1, c0, c1):
    H, W = len(mapping), len(mapping[0])
    r0 = max(0, int(r0)); r1 = min(H, int(r1))
    c0 = max(0, int(c0)); c1 = min(W, int(c1))
    out = set()
    for y in range(r0, r1):
        row = mapping[y]
        for x in range(c0, c1):
            out.update(row[x])
    return out

def extract_cells_from_bbox(mapping, bbox, include_boundary=True):
    x_min, y_min, x_max, y_max = bbox
    if include_boundary:
        c0 = np.floor(x_min); c1 = np.ceil(x_max)
        r0 = np.floor(y_min); r1 = np.ceil(y_max)
    else:
        c0 = np.ceil(x_min);  c1 = np.floor(x_max)
        r0 = np.ceil(y_min);  r1 = np.floor(y_max)
    return extract_cells_from_matrix_window(mapping, r0, r1, c0, c1)

def extract_cells_by_indices(adata, indices):
    mask = np.zeros(adata.n_obs, dtype=bool)
    if len(indices) > 0:
        mask[list(indices)] = True
    return adata[mask]



def normalize_spatial(spatial: np.ndarray, per_axis: bool = True, eps: float = 1e-12):
    spatial = np.asarray(spatial, dtype=np.float64)
    assert spatial.ndim == 2 and spatial.shape[1] == 2, "spatial should be (N,2)"

    center = (spatial.min(axis=0) + spatial.max(axis=0)) / 2.0
    centered = spatial - center

    max_abs = np.maximum(np.abs(centered).max(axis=0), eps)  
    if per_axis:
        scale = max_abs
    else:
        s = float(np.max(max_abs))
        scale = np.array([s, s], dtype=np.float64)

    spatial_norm = centered / scale
    spatial_norm = np.clip(spatial_norm, -1.0, 1.0)

    return spatial_norm, {"center": center, "scale": scale}


def denormalize_spatial(spatial_norm: np.ndarray, params: dict):
    spatial_norm = np.asarray(spatial_norm, dtype=np.float64)
    center = np.asarray(params["center"], dtype=np.float64)
    scale  = np.asarray(params["scale"], dtype=np.float64)
    return spatial_norm * scale + center




def rotate_spatial(spatial: np.ndarray, angle_degrees: float, center: Optional[np.ndarray] = None):
    spatial = np.asarray(spatial, dtype=np.float64)
    assert spatial.ndim == 2 and spatial.shape[1] == 2, "spatial should be (N,2)"

    if center is None:
        X = spatial
    else:
        center = np.asarray(center, dtype=np.float64)
        X = spatial - center

    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s,  c]], dtype=np.float64)

    Y = X @ R.T
    if center is not None:
        Y = Y + center
    return Y



def normalized_rotation_perturbation(
    spatial: np.ndarray,
    angle_degrees: float,
    per_axis: bool = True
):
    spatial_norm, params = normalize_spatial(spatial, per_axis=per_axis)
    spatial_rot = rotate_spatial(spatial_norm, angle_degrees, center=None)      
    return spatial_rot, params