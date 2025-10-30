import numpy as np
import matplotlib.pyplot as plt


def suggest_joint_square_canvas_two(coords1, coords2,
                                    target_mean_occupancy=1.0,
                                    min_side=32, max_side=2048,
                                    gap_pixels=0.5):
    if gap_pixels and gap_pixels > 0:
        r_target = 1.0 / float(gap_pixels + 1)**2
        lam = -np.log(max(1.0 - r_target, 1e-12))  
    else:
        lam = float(target_mean_occupancy)

    def _suggest_one(coords):
        coords = np.asarray(coords, dtype=float)
        N = coords.shape[0]
        xmin, xmax = coords[:,0].min(), coords[:,0].max()
        ymin, ymax = coords[:,1].min(), coords[:,1].max()
        dx = max(xmax - xmin, 1e-8)
        dy = max(ymax - ymin, 1e-8)
        aspect = dx / dy  

        total_pixels = max(N / lam, 1.0)  
        W = int(np.clip(np.sqrt(total_pixels * aspect), min_side, max_side))
        H = int(np.clip(np.sqrt(total_pixels / aspect), min_side, max_side))
        return H, W

    H1, W1 = _suggest_one(coords1)
    H2, W2 = _suggest_one(coords2)

    P1, P2 = H1 * W1, H2 * W2
    Hc, Wc = (H1, W1) if P1 <= P2 else (H2, W2)  
    S = int(np.clip(max(Hc, Wc), min_side, max_side))
    return (S, S)   

def create_rgb_image_with_mapping(coords, rgb_values, img_size=(100, 100)):
    coords = np.asarray(coords, dtype=float)
    vals   = np.asarray(rgb_values)           
    H, W   = int(img_size[0]), int(img_size[1])
    C      = int(vals.shape[1])

    img = np.zeros((H, W, C), dtype=vals.dtype)
    pixel_mapping = [[[] for _ in range(W)] for _ in range(H)]

    x_scaled = np.interp(coords[:, 0], (coords[:, 0].min(), coords[:, 0].max()), (0, W - 1)).astype(int)
    y_scaled = np.interp(coords[:, 1], (coords[:, 1].min(), coords[:, 1].max()), (0, H - 1)).astype(int)
    x_scaled = np.clip(x_scaled, 0, W - 1)
    y_scaled = np.clip(y_scaled, 0, H - 1)

    for idx, (x, y, v) in enumerate(zip(x_scaled, y_scaled, vals)):
        img[y, x] = v                  
        pixel_mapping[y][x].append(idx)

    return img, pixel_mapping

def interpolate_zero_pixels(
    img: np.ndarray,
    radius: int = 1,
    min_neighbors: int = 3,
    ring_only: bool = False,
    aggregate: str = "mean",
):

    if radius < 1:
        raise ValueError("radius must >= 1")
    if aggregate not in {"mean", "median"}:
        raise ValueError("aggregate only support 'mean' or 'median'")

    H, W, C = img.shape
    out = img.copy()

    offsets = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            if ring_only:
                if max(abs(di), abs(dj)) == radius:
                    offsets.append((di, dj))
            else:
                if max(abs(di), abs(dj)) <= radius:
                    offsets.append((di, dj))

    is_zero = np.all(img == 0, axis=2)

    for i in range(H):
        for j in range(W):
            if not is_zero[i, j]:
                continue

            buf = []
            for di, dj in offsets:
                ii, jj = i + di, j + dj
                if 0 <= ii < H and 0 <= jj < W:
                    if not np.all(img[ii, jj] == 0):
                        buf.append(img[ii, jj])

            if len(buf) >= min_neighbors:
                neigh = np.asarray(buf, dtype=np.float32)
                if aggregate == "mean":
                    val = neigh.mean(axis=0)
                else:
                    val = np.median(neigh, axis=0)
                out[i, j] = val.astype(out.dtype, copy=False)

    return out


def find_cell_positions_in_image(coords, img_size=(100, 100)):
    coords = np.asarray(coords, dtype=float)
    H, W = int(img_size[0]), int(img_size[1])

    x_scaled = np.interp(coords[:, 0], (coords[:, 0].min(), coords[:, 0].max()), (0, W - 1)).astype(int)
    y_scaled = np.interp(coords[:, 1], (coords[:, 1].min(), coords[:, 1].max()), (0, H - 1)).astype(int)
    x_scaled = np.clip(x_scaled, 0, W - 1)
    y_scaled = np.clip(y_scaled, 0, H - 1)

    cell_positions = np.column_stack([x_scaled, y_scaled]).astype(int)
    return cell_positions



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