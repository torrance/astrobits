from __future__ import division

from numba import njit, float32, float64, prange
import numpy as  np

# This is valid in the flux region 10^-7.5 < S < 75 Jy

@njit([float32[:](float32[:]), float64[:](float64[:])], parallel=True)
def logdNdS(logS):
    _logdNdS = np.empty_like(logS)

    # Split flux into two domains, one using Franzen 2019, the other our own parameterisation
    # calculated from SKADS model in Wilham 2008.
    for idx in prange(0, len(logS)):
        if logS[idx] > -3:
            _logdNdS[idx] = (
                + 3.52
                + 0.307 * logS[idx]
                - 0.388 * logS[idx]**2
                - 0.0404 * logS[idx]**3
                + 0.0351 * logS[idx]**4
                + 0.006 * logS[idx]**5
                - 2.5 * logS[idx]
            )
        else:
            _logdNdS[idx] = (
                - 1.21710946e+01
                - 1.34627966e+01 * logS[idx]
                - 5.01013972e+00 * logS[idx]**2
                - 9.08609418e-01 * logS[idx]**3
                - 8.44882862e-02 * logS[idx]**4
                - 3.13722304e-03 * logS[idx]**5
                - 2.5 * logS[idx]
            )

    return _logdNdS


@njit([float32[:](float32[:]), float64[:](float64[:])])
def dNdS(S):
    logS = np.log10(S)
    return 10**logdNdS(logS)


