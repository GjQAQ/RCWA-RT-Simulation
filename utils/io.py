from pathlib import Path

import numpy as np
import scipy

__all__ = [
    'load_rcwa_data',
]


def load_rcwa_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)

    k = scipy.io.loadmat(str(path / 'k.mat'))['results']
    t = scipy.io.loadmat(str(path / 't.mat'))['results']
    return k, t
