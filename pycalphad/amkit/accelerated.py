"""
Accelerated numerical kernels with Numba JIT and NumPy fallbacks.
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.jit(nopython=True, fastmath=True)
    def _fvm_3d_laplacian_numba(T, dx2, dy2, dz2):
        nx, ny, nz = T.shape
        d2T = np.zeros_like(T)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    d2T[i, j, k] = (
                        (T[i+1, j, k] - 2*T[i, j, k] + T[i-1, j, k]) / dx2 +
                        (T[i, j+1, k] - 2*T[i, j, k] + T[i, j-1, k]) / dy2 +
                        (T[i, j, k+1] - 2*T[i, j, k] + T[i, j, k-1]) / dz2
                    )
        return d2T
else:
    def _fvm_3d_laplacian_numba(T, dx2, dy2, dz2):
        d2T = np.zeros_like(T)
        d2T[1:-1, 1:-1, 1:-1] = (
            (T[2:, 1:-1, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[:-2, 1:-1, 1:-1]) / dx2 +
            (T[1:-1, 2:, 1:-1] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, :-2, 1:-1]) / dy2 +
            (T[1:-1, 1:-1, 2:] - 2*T[1:-1, 1:-1, 1:-1] + T[1:-1, 1:-1, :-2]) / dz2
        )
        return d2T

def compute_3d_laplacian(T: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Computes 3D FVM Laplacian using Numba JIT if available, or fast NumPy slice vectorization.
    """
    return _fvm_3d_laplacian_numba(T, dx**2, dy**2, dz**2)
