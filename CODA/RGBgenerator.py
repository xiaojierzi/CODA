import numpy as np
import matplotlib.pyplot as plt

class RGBImageGenerator:
    def __init__(self, img_size=(100, 100)):
        self.img_size = img_size

    def create_rgb_image_with_mapping(self, coords, rgb_values):
        img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        pixel_mapping = [[[] for _ in range(self.img_size[1])] for _ in range(self.img_size[0])]

        x_scaled = np.interp(coords[:, 0], (coords[:, 0].min(), coords[:, 0].max()), (0, self.img_size[0]-1)).astype(int)
        y_scaled = np.interp(coords[:, 1], (coords[:, 1].min(), coords[:, 1].max()), (0, self.img_size[1]-1)).astype(int)

        for idx, (x, y, color) in enumerate(zip(x_scaled, y_scaled, rgb_values)):
            img[y, x] = color
            pixel_mapping[y][x].append(idx)
        
        return img, pixel_mapping

    def interpolate_zero_pixels(self, img, ring_size=1):

        height, width, _ = img.shape
        img_interpolated = img.copy()
        neighbors_offsets = self.get_neighbors_offsets(ring_size)
        
        for i in range(ring_size, height - ring_size):
            for j in range(ring_size, width - ring_size):
                if np.all(img[i, j] == 0):
                    neighbor_values = [img[i + off[0], j + off[1]] for off in neighbors_offsets if not np.all(img[i + off[0], j + off[1]] == 0)]
                    if len(neighbor_values) >= 3:
                        img_interpolated[i, j] = np.mean(neighbor_values, axis=0)
        
        return img_interpolated

    def get_neighbors_offsets(self, ring_size):
        return [(i, j) for i in range(-ring_size, ring_size + 1) for j in range(-ring_size, ring_size + 1) if i != 0 or j != 0]

    def show_image(self, img, title="Image"):
        plt.imshow(img)
        plt.title(title)
        plt.show()