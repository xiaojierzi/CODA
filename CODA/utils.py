import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as IPyImage, display

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
