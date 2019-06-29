from __future__ import print_function, division

from astropy.coordinates import SkyCoord
from astropy.io.fits import getheader
from astropy.time import Time
import astropy.units as units
import mwa_pb.config
import mwa_pb.primary_beam as pb
from numba import njit
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

    def jones(self, ras, decs, freqs):
        jones = np.zeros((len(ras), len(freqs), 2, 2), dtype=np.complex)
        for i, freq in enumerate(freqs):
            alt, az = radec_to_altaz(ras, decs, self.time, self.location)
            jones[:, i] = pb.MWA_Tile_full_EE(np.pi/2 - alt, az, freq, delays=self.delays, jones=True)

        return jones


@njit()
def _minimalmap(ras, decs, radius):
    N = len(ras)
    Nmatched = 0
    left = np.zeros(N, dtype=np.int64)
    right = np.zeros(N, dtype=np.int64)
    matched = np.zeros(N, dtype=np.int64)
    dtheta = np.empty(N, dtype=np.float64)

    sindecs = np.sin(decs)
    cosdecs = np.cos(decs)

    for i in range(N):
        if i % 1000 == 0:
            print(i)

        if matched[i]:
            continue

        # Find all coordinates within some radius of our seed coordinate
        dtheta[:] = np.inf
        dtheta[~matched] = np.arccos(
            sindecs[i] * sindecs[~matched] +
            cosdecs[i] * cosdecs[~matched] * np.cos(np.absolute(ras[~matched] - ras[i]))
        )
        dtheta[i] = 0  # Special case; sometimes this gives nan
        nearby = (dtheta < radius).nonzero()[0]

        # Link all nearby coordinates to seed
        n = len(nearby)
        left[Nmatched:Nmatched + n] = i
        right[Nmatched:Nmatched + n] = nearby
        matched[nearby] = True
        Nmatched += n

    assert(N == Nmatched)
    return (left, right)

