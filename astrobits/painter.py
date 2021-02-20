from __future__ import print_function, division

from numba import njit, float64
import numpy as np


@njit()
def paint(canvas, xs, ys, vals):
    """
    Paints values at xs, ys onto canvas.

    Adheres to fits convention of canvas[y, x] NOT C convention of canvas[x, y].

    If xs, ys are not integers, this will distribute the val across 4 pixels weighted by
    the excess float.

    An (x, y) value corresponds to the _center_ of a pixel.
    """

    weights = np.zeros((2, 2), dtype=float64)

    for x, y, val in zip(xs, ys, vals):
        x_int, y_int = int(x), int(y)

        # Handle special case when x and y are exactly in the pixel center
        if x == x_int and y == y_int:
            if 0 <= x_int < canvas.shape[0] and 0 <= y_int < canvas.shape[1]:
                canvas[y_int, x_int] += val
            continue

        # Calculate proportionate weights for the 4 nearby pixels
        for i in [0, 1]:
            for j in [0, 1]:
                xi, yj = x_int + i, y_int + j
                weights[i, j] = 1 / np.sqrt((x - xi)**2 + (y - yj)**2)

        weights /= weights.sum()

        # Set flux weighted by nearby pixels
        for i in [0, 1]:
            for j in [0, 1]:
                xi, yj = x_int + i, y_int + j
                if 0 <= yj < canvas.shape[0] and 0 <= xi < canvas.shape[1]:
                    canvas[yj, xi] += weights[i, j] * val

    return canvas
