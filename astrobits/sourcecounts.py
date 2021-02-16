from __future__ import division

from numba import njit, float32, float64, prange
import numpy as  np

# This is valid in the flux region 10^-7.5 < S < 75 Jy at 154 MHz.

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


@njit([float32[:](float32[:]), float64[:](float64[:])])
def dNdSTRECS1400(Ss):
    """
    This is a fitted dNdS to TRECS deep1x1 simulation at 1400 MHz.
    It is fit on data from 1E-1 down to 1E-9.
    """
    coeffs = [-5.61548545e-05, -2.83866491e-03, -5.48423751e-02, -5.28207741e-01, -2.71030259e+00, -7.28522125e+00, -8.81332290e+00, -1.74312446e+00]

    logSs = np.log10(Ss)
    logdNdSTRECS = np.zeros_like(logSs)
    logdNdSTRECS += -2.5 * logSs

    for power, coeff in enumerate(coeffs[::-1]):
        logdNdSTRECS += coeff * logSs**power

    return 10**logdNdSTRECS


@njit([float32[:](float32[:]), float64[:](float64[:])])
def dNdSTRECS150(Ss):
    """
    This is a fitted dNdS to TRECS deep1x1 simulation at 150 MHz.
    It is fit on data from 1E-0 down to 1E-8.5.
    """
    coeffs = [-8.26643173e-05, -3.10493654e-03, -4.63622889e-02, -3.44862848e-01, -1.30157946e+00, -2.27034046e+00, -8.54033001e-01,  3.33331939e+00]

    logSs = np.log10(Ss)
    logdNdSTRECS = np.zeros_like(logSs)
    logdNdSTRECS += -2.5 * logSs

    for power, coeff in enumerate(coeffs[::-1]):
        logdNdSTRECS += coeff * logSs**power

    return 10**logdNdSTRECS
