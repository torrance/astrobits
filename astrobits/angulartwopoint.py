from __future__ import print_function, division

from multiprocessing import Pool
import sys

import numpy as np
from scipy.spatial import cKDTree


def angular_two_point(data, bins, N = 100, Rn = None, datafilter=None):
    print("Calculating angular correlation function...", end="", file=sys.stderr)
    sys.stdout.flush()

    # Set Rn default to the same size as D
    Rn = data.shape[1] if Rn is None else Rn

    # Convert angular bin values to cartesian distances
    origin = radec_to_xyz(0, 0)  # [x, y, z]
    bins = radec_to_xyz(np.zeros_like(bins), bins)  # [x, y, z]
    bins = np.sqrt(np.sum((bins - origin[:, None])**2, axis=0))  # [r]

    min_ra, max_ra, min_dec, max_dec = min(data[0]), max(data[0]), min(data[1]), max(data[1])

    # Convert angular data points to cartesian points
    data = radec_to_xyz(data[0], data[1])
    D = cKDTree(data.T)
    DD = D.count_neighbors(D, r=bins, cumulative=False)[1:]

    pool = Pool()
    results = []
    corrs = np.empty((2, N, len(bins) - 1))
    for _ in range(N):
        res = pool.apply_async(
            _angular_two_point,
            (cKDTree(data.T), bins, min_ra, max_ra, min_dec, max_dec, Rn, DD, datafilter),
        )
        results.append(res)
    pool.close()
    pool.join()
    for i, res in enumerate(results):
        corrs[:, i, :] = res.get()

    print("Done", file=sys.stderr)

    return corrs


def _angular_two_point(D, bins, min_ra, max_ra, min_dec, max_dec, Rn, DD, datafilter):
    # Each worker thread needs a new seed, or else they'll generate the same random values
    np.random.seed()

    # Generate random sample
    R = uniform_sphere(min_ra, max_ra, min_dec, max_dec, Rn)
    # Allow for bespoke masking of region
    if datafilter:
        R = datafilter(R)
    R = radec_to_xyz(R[0], R[1])  # [x, y, z]
    R = cKDTree(R.T)

    # DD = D.count_neighbors(D, r=bins, cumulative=False)[1:]
    DR = D.count_neighbors(R, r=bins, cumulative=False)[1:]
    RR = R.count_neighbors(R, r=bins, cumulative=False)[1:]
    f = D.n / R.n

    corr = np.zeros((2, len(bins) - 1))
    corr[0] = (DD - 2 * f * DR + f**2 * RR) / (f**2 * RR)
    corr[1] = (DD * RR) / (DR * DR) - 1

    print(".", end="", file=sys.stderr)
    sys.stdout.flush()

    return corr


def uniform_sphere(ra_min, ra_max, dec_min, dec_max, N):
    z_min = np.sin(dec_min)
    z_max = np.sin(dec_max)
    zs = np.random.uniform(z_min, z_max, N)
    decs = np.arcsin(zs)

    ras = np.random.uniform(ra_min, ra_max, N)

    return np.array((ras, decs))


def radec_to_xyz(ra, dec):
    phi = ra
    theta = np.pi / 2 - dec

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array((x, y, z))
