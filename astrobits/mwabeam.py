from __future__ import print_function, division

from multiprocessing.dummy import Pool
import threading
import time as tm
import sys

from astropy.coordinates import SkyCoord
from astropy.io.fits import getheader
from astropy.time import Time
import astropy.units as units
import mwa_pb.config
import mwa_pb.beam_full_EE as beam_full_EE
import mwa_pb.primary_beam as pb
from numba import njit, float64, int64, prange
import numpy as np

from astrobits.coordinates import radec_to_altaz


# Global threadlock to ensure threadsafe access to mwa_pb
threadlock = threading.Lock()


class MWABeam(object):
    def __init__(self, metafits):
        # Open metafits and extract beam delays
        metafits = getheader(metafits)
        self.time = Time(metafits['DATE-OBS'], location=mwa_pb.config.MWAPOS)
        delays = [int(d) for d in metafits['DELAYS'].split(',')]
        self.delays = [delays, delays] # Shuts up mwa_pb
        self.location = mwa_pb.config.MWAPOS

    def jones(self, ras, decs, freq, time=None):
        if time is None:
            time = self.time

        alt, az = radec_to_altaz(ras, decs, time, self.location)
        with threadlock:
            return pb.MWA_Tile_full_EE(np.pi/2 - alt, az, freq, delays=self.delays, jones=True)

    def joness(self, ras, decs, freq, time=None):
        if time is None:
            time = self.time

        t0 = tm.time()
        alt, az = radec_to_altaz(ras, decs, time, self.location)
        za = np.pi / 2 - alt

        # Calculate Jones vector across grid
        with threadlock:
            tile = beam_full_EE.get_AA_Cached(target_freq_Hz=freq)
            beam = beam_full_EE.Beam(tile, self.delays, amps=np.ones([2, 16]))
            grid_zas = np.radians(np.linspace(0, 90, 5 * 90 + 1))
            grid_azs = np.radians(np.linspace(0, 360, 5 * 360 + 1))

            gridded_jones = beam.get_FF(grid_azs, grid_zas, grid=True)  # [2, 2, alt, az]
            gridded_jones = np.transpose(gridded_jones, [2, 3, 0, 1])  # [alt, az, 2, 2]
        print("Gridding beam elapsed: %g" % (tm.time() - t0)); sys.stdout.flush()

        # Interpolate Jones vector onto our points
        jones = np.zeros((len(ras), 2, 2), dtype=np.complex)

        def _thread(start, end):
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                jones[start:end, i, j] += RegularGridInterpolator(
                    (grid_azs, grid_zas),
                    gridded_jones[:, :,  i, j].real,
                    method='linear',
                    bounds_error=False,
                    fill_value=0,
                )(np.array((az[start:end], za[start:end])).T)

                jones[start:end, i, j] += 1j * RegularGridInterpolator(
                    (grid_azs, grid_zas),
                    gridded_jones[:, :,  i, j].imag,
                    method='linear',
                    bounds_error=False,
                    fill_value=0,
                )(np.array((az[start:end], za[start:end])).T)

        t0 = tm.time()
        batch = 100000
        pool = Pool()
        for start in range(0, len(ras), batch):
            end = start + batch
            pool.apply_async(_thread, (start, end))
            #_thread(start, end)

        pool.close()
        pool.join()
        print("Interpolating beam points elapsed: %g" % (tm.time() - t0)); sys.stdout.flush()

        return jones


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

