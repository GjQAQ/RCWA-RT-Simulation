import matplotlib.pyplot as plt
import numpy as np


def vis_rcwa_t(rcwa_t):
    n_order = rcwa_t.shape[-2]
    n_p = rcwa_t.shape[-1]
    fig, ax = plt.subplots(n_order, n_p * 2, figsize=(3.2 * n_p, 3.2 * n_order))

    for i in range(n_order):
        for j in range(n_p):
            ax[i, j * 2].imshow(np.abs(rcwa_t[..., i, j]), cmap='inferno')
            ax[i, j * 2].set(xticks=[], yticks=[])
            ax[i, j * 2 + 1].imshow(np.angle(rcwa_t[..., i, j]), cmap='hsv')
            ax[i, j * 2 + 1].set(xticks=[], yticks=[])
