from __future__ import division, print_function

import sys

from numba import njit
import numpy as np
from scipy.signal import correlate


def radialcrosscorrelation(img, kernel, bins, normalize=True, imgmean=None, kernelmean=None, sigma2=None):
    assert(img.shape == kernel.shape)

    if normalize:
        # Calculate Ns
        Ns = np.ones_like(img)
        print("Calculating Ns...", end=""); sys.stdout.flush()
        Ns = correlate(Ns, Ns, mode="same")
        print(" Done.")

    # Calculate autocorrelation
    if imgmean is None:
        img =  img - np.mean(img)
    else:
        img = img - imgmean

    if kernelmean is None:
        kernel = kernel - np.mean(kernel)
    else:
        kernel = kernel - kernelmean

    if sigma2 is None:
        sigma2 = np.std(img) * np.std(kernel)

    # In the case of an infinite img (or an image that reaches zero from the edges to infinity)
    # this is the correct normalisation.
    # auto = correlate(img, img, mode="same") / (img.size * sigma**2)
    # Otherwise, we have to divide by the autocorrelation of the 'primary beam'
    print("Calculating correlation...", end=""); sys.stdout.flush()
    auto = correlate(img, kernel, mode="same") # / (Ns * sigma2)
    if normalize:
        auto /= (Ns * sigma2)
    print(" Done.")

    # Radial average
    dists = calcdists(img.shape)
    idxs = np.digitize(dists, bins)
    mus, sigmas = radialaverage(auto, bins, idxs)

    # Also output the raw values prior to radial averaging
    idxs = np.argsort(dists, axis=None)

    return mus, sigmas, dists.reshape(-1)[idxs], auto.reshape(-1)[idxs]


def radialautocorrelation(img, bins):
    return radialcrosscorrelation(img, img, bins)


def calcdists(shape):
    Xs, Ys = np.meshgrid(range(shape[0]), range(shape[1]))
    midX = shape[0] // 2
    midY = shape[1] // 2
    return np.sqrt((Xs - midX)**2 + (Ys - midY)**2)


@njit()
def radialaverage(auto, bins, idxs):
    auto = auto.reshape(-1)
    idxs = idxs.reshape(-1)

    mus = np.zeros(len(bins) - 1)
    variances = np.zeros(len(bins) - 1)
    Ns = np.zeros(len(bins) - 1)

    for val, idx in zip(auto, idxs):
        if 0 < idx < len(bins):
            mus[idx - 1] += val
            Ns[idx - 1] += 1

    mus /= Ns

    for val, idx in zip(auto, idxs):
        if 0 < idx < len(bins):
            variances[idx] += (val - mus[idx])**2

    variances /= Ns
    sigmas = np.sqrt(variances)

    return mus, sigmas

# def return np.digitize(dists, bins)
