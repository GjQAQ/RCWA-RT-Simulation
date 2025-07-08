import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'vis_direction',
]

A = np.ndarray


def _axis(n):
    if n == 1:
        return np.array([0.])
    else:
        return np.linspace(-1, 1, n)


def vis_direction(d: A, orders: int | tuple[int, int]):
    if isinstance(orders, int):
        orders = (orders, 1)

    if d.shape[-1] != 3:
        raise ValueError('direction must be 3D')
    d = np.reshape(d, orders + (3,))

    x = _axis(orders[0])
    y = _axis(orders[1])
    x, y = np.meshgrid(x, y, indexing='ij')
    z = np.zeros_like(x)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.quiver(x, y, z, d[..., 0], d[..., 1], d[..., 2], arrow_length_ratio=0.1)

    ax.fill_between([-1, -1], [-1, 1], [0, 0], [1, 1], [-1, 1], [0, 0], alpha=0.5)  # noqa

    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set(xticks=ticks, yticks=ticks, zticks=ticks)
    scale = 1
    ax.set(xlim=(-scale, scale), ylim=(-scale, scale), zlim=(-scale, scale))

    return fig, ax
