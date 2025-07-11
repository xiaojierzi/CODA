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
from matplotlib.patches import Rectangle, Polygon
from scipy.spatial import ConvexHull
from shapely.geometry import Point




def direct_image_mapping_with_coords(img, phi, pixel_mapping):

    mapped_img = np.zeros_like(img)
    
    max_cell_index = max(max(indices) for row in pixel_mapping for indices in row if indices)
    new_cell_coords = np.zeros((max_cell_index + 1, 2)) 
    
    for src_y in range(img.shape[0]):
        for src_x in range(img.shape[1]):
            target_y, target_x = phi[src_x, src_y]
            target_x = int(np.round(target_x))
            target_y = int(np.round(target_y))
            target_x = np.clip(target_x, 0, img.shape[1] - 1)
            target_y = np.clip(target_y, 0, img.shape[0] - 1)
            mapped_img[target_y, target_x] = img[src_y, src_x]
            
            for cell_index in pixel_mapping[src_y][src_x]:
                new_cell_coords[cell_index] = [target_x, target_y]

    return mapped_img, new_cell_coords


def find_cell_positions_in_image(coords, img_size=(100, 100)):

    cell_positions = np.zeros((coords.shape[0], 2), dtype=int)

    x_scaled = np.interp(coords[:, 0], (coords[:, 0].min(), coords[:, 0].max()), (0, img_size[0] - 1)).astype(int)
    y_scaled = np.interp(coords[:, 1], (coords[:, 1].min(), coords[:, 1].max()), (0, img_size[1] - 1)).astype(int)

    cell_positions[:, 0] = x_scaled
    cell_positions[:, 1] = y_scaled

    return cell_positions


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
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
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
    mode: str = "bbox",  # "bbox" or "convex"
    lw: float = 2.0
):

    kpts0_np = kpts0.cpu().numpy()
    kpts1_np = kpts1.cpu().numpy()

    axes = plot_images([image0, image1])
    plot_matches(kpts0, kpts1, color="lime", lw=0.2)

    if mode == "bbox":
        x0_min, x0_max = kpts0_np[:, 0].min(), kpts0_np[:, 0].max()
        y0_min, y0_max = kpts0_np[:, 1].min(), kpts0_np[:, 1].max()
        x1_min, x1_max = kpts1_np[:, 0].min(), kpts1_np[:, 0].max()
        y1_min, y1_max = kpts1_np[:, 1].min(), kpts1_np[:, 1].max()

        plot_rect(axes[0], x0_min, x0_max, y0_min, y0_max, color="red", lw=lw)
        plot_rect(axes[1], x1_min, x1_max, y1_min, y1_max, color="blue", lw=lw)

    elif mode == "convex":
        def plot_hull(ax, points, color):
            if len(points) >= 3:
                hull = ConvexHull(points)
                polygon = Polygon(points[hull.vertices], closed=True, edgecolor=color, fill=False, linewidth=lw)
                ax.add_patch(polygon)

        plot_hull(axes[0], kpts0_np, color="red")
        plot_hull(axes[1], kpts1_np, color="blue")

    else:
        raise ValueError(f"Unsupported mode: {mode}, use 'bbox' or 'convex'.")

    kpc0 = cm_prune(prune0)
    kpc1 = cm_prune(prune1)

    return axes

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


def extract_cells_by_indices(adata, indices):
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[list(indices)] = True
    return adata[mask]