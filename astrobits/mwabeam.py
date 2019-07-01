from __future__ import print_function, division

from astropy.coordinates import SkyCoord
from astropy.io.fits import getheader
from astropy.time import Time
import astropy.units as units
import mwa_pb.config
import mwa_pb.primary_beam as pb
from numba import njit, float64, int64, prange
import numpy as np

from astrobits.coordinates import radec_to_altaz


class MWABeam(object):
    def __init__(self, metafits):
        # Open metafits and extract beam delays
        metafits = getheader(metafits)
        self.time = Time(metafits['DATE-OBS'], location=mwa_pb.config.MWAPOS)
        delays = [int(d) for d in metafits['DELAYS'].split(',')]
        self.delays = [delays, delays] # Shuts up mwa_pb
        self.location = mwa_pb.config.MWAPOS

    def jones(self, ras, decs, freq):
        alt, az = radec_to_altaz(ras, decs, self.time, self.location)
        return pb.MWA_Tile_full_EE(np.pi/2 - alt, az, freq, delays=self.delays, jones=True)


def minimalmap(ras, decs, radius):
    coords = SkyCoord(ras, decs, unit=(units.radian, units.radian))
    idx1, idx2, _, _ = coords.search_around_sky(coords, radius)

    mapped = _minimalmap(len(ras), idx1, idx2)
    return np.unique(mapped, return_inverse=True)  # seeds, inverse


@njit(parallel=True)
def _minimalmap(N, idx1, idx2):
    seeds = np.zeros(N, dtype=np.int64)
    mapped = np.empty(N, dtype=np.int64)
    mapped[:] = -1

    for i in range(len(idx1)):
        # if i % 1000 == 0:
        #     print(i, len(idx1))

        i1, i2 = idx1[i], idx2[i]
        if mapped[i2] != -1:
            # Mapping already exists
            continue
        elif seeds[i1]:
            # i1 is a seed already
            mapped[i2] = i1
        else:
            # Set up i1 as a new seed
            seeds[i1] = True
            mapped[i1] = i1
            mapped[i2] = i1

    return mapped



# @njit([int64[:](float64[:], float64[:], float64)], parallel=True)
# def _minimalmap(ras, decs, radius):
#     N = len(ras)
#     Nseeds = 1
#     seeds = np.zeros(N, dtype=np.int64)
#     mapped = np.zeros(N, dtype=np.int64)
#     cosdthetas = np.zeros(N, dtype=np.float64)

#     # Precompute what we can
#     cosradius = np.cos(radius)
#     sindecs = np.sin(decs)
#     cosdecs = np.cos(decs)
#     sinras = np.sin(ras)
#     cosras = np.cos(ras)

#     # Bootstrap first seed
#     seeds[0] = 0

#     for i in range(1, N):
#         if i % 1000 == 0:
#             print(i, Nseeds)

#         # Calculate angular distance to existing seeds
#         _seeds = seeds[:Nseeds]
#         for j in prange(Nseeds):
#             cosdthetas[j] = (
#                 sindecs[i] * sindecs[_seeds][j] +
#                 cosdecs[i] * cosdecs[_seeds][j] * (
#                     sinras[i] * sinras[_seeds][j] + cosras[i] * cosras[_seeds][j]
#                 )
#             )
#         idxmax = np.argmax(cosdthetas[:Nseeds])

#         if cosdthetas[idxmax] > cosradius:
#             # Associate point with seed
#             mapped[i] = _seeds[idxmax]
#         else:
#             # Ceate new seed and associate with self
#             seeds[Nseeds] = i
#             Nseeds += 1
#             mapped[i] = i

#     return mapped


# @njit()
# def _minimalmap(ras, decs, radius):
#     N = len(ras)
#     Nmatched = 0
#     left = np.zeros(N, dtype=np.int64)
#     right = np.zeros(N, dtype=np.int64)
#     matched = np.zeros(N, dtype=np.int64)
#     dtheta = np.empty(N, dtype=np.float64)

#     sindecs = np.sin(decs)
#     cosdecs = np.cos(decs)

#     for i in range(N):
#         if i % 1000 == 0:
#             print(i)

#         if matched[i]:
#             continue

#         # Find all coordinates within some radius of our seed coordinate
#         dtheta[:] = np.inf
#         dtheta[~matched] = np.arccos(
#             sindecs[i] * sindecs[~matched] +
#             cosdecs[i] * cosdecs[~matched] * np.cos(np.absolute(ras[~matched] - ras[i]))
#         )
#         dtheta[i] = 0  # Special case; sometimes this gives nan
#         nearby = (dtheta < radius).nonzero()[0]

#         # Link all nearby coordinates to seed
#         n = len(nearby)
#         left[Nmatched:Nmatched + n] = i
#         right[Nmatched:Nmatched + n] = nearby
#         matched[nearby] = True
#         Nmatched += n

#     assert(N == Nmatched)
#     return (left, right)

