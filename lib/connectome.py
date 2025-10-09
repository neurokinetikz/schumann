import numpy as np


def compute_connectome_harmonics(adj_matrix: np.ndarray, laplacian: str = 'normalized'):
    adj_matrix = np.asarray(adj_matrix, float)
    degree = np.diag(adj_matrix.sum(axis=1))
    if laplacian == 'normalized':
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree.diagonal(), 1e-12)))
        L = np.eye(adj_matrix.shape[0]) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    else:
        L = degree - adj_matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    return eigvals[idx], eigvecs[:, idx]


def project_time_series_onto_harmonics(time_series: np.ndarray, eigvecs: np.ndarray):
    time_series = np.asarray(time_series, float)
    proj = eigvecs.T @ time_series
    return proj
