import scanpy as sc
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import matplotlib.pyplot as plt

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