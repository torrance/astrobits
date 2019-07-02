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
import numpy as np
from scipy.interpolate import RegularGridInterpolator

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
        self.rgi_cache = {}

    def jones(self, ras, decs, freq, time=None):
        if time is None:
            time = self.time

        t0 = tm.time()
        alt, az = radec_to_altaz(ras, decs, time, self.location)
        print("Altaz elapsed: %g" % (tm.time() - t0))
        with threadlock:
            return pb.MWA_Tile_full_EE(np.pi/2 - alt, az, freq, delays=self.delays, jones=True)

    def joness(self, ras, decs, freqs, time=None):
        if time is None:
            time = self.time

        t0 = tm.time()
        alt, az = radec_to_altaz(ras, decs, time, self.location)
        za = np.pi / 2 - alt
        print("Altaz elapsed: %f" % (tm.time() - t0))

        # Interpolate Jones vector onto our points
        jones = np.zeros((len(ras), len(freqs), 2, 2), dtype=np.complex)

        def _thread(k, freq):
            rgi = self.get_rgi(freq)

            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                jones[:, k, i, j] += rgi[i][j][0](
                    np.array((az, za)).T
                )

                jones[:, k, i, j] += 1j * rgi[i][j][1](
                    np.array((az, za)).T
                )

        t0 = tm.time()
        pool = Pool()
        for k, freq in enumerate(freqs):
            pool.apply_async(_thread, (k, freq))
            #_thread(k, freq)

        pool.close()
        pool.join()
        print("Interpolating beam points elapsed: %g" % (tm.time() - t0)); sys.stdout.flush()

        return jones

    def get_rgi(self, freq):
        # Calculate Jones vector across grid
        try:
            # Try to see if we've cached the interpolator already
            return self.rgi_cache[freq]
        except KeyError:
            with threadlock:
                grid_zas = np.radians(np.linspace(0, 90, 10 * 90 + 1))
                grid_azs = np.radians(np.linspace(0, 360, 10 * 360 + 1))

                tile = beam_full_EE.get_AA_Cached(target_freq_Hz=freq)
                beam = beam_full_EE.Beam(tile, self.delays, amps=np.ones([2, 16]))

                gridded_jones = beam.get_FF(grid_azs, grid_zas, grid=True)  # [2, 2, alt, az]
                gridded_jones = tile.apply_zenith_norm_Jones(gridded_jones)  # Normalise to Zenith
                gridded_jones = np.transpose(gridded_jones, [2, 3, 0, 1])  # [alt, az, 2, 2]

                rgi = [[[], []], [[], []]]
                for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    rgi[i][j].append(RegularGridInterpolator(
                        (grid_azs, grid_zas),
                        gridded_jones[:, :,  i, j].real,
                        method='linear',
                        bounds_error=False,
                        fill_value=0,
                    ))
                    rgi[i][j].append(RegularGridInterpolator(
                        (grid_azs, grid_zas),
                        gridded_jones[:, :,  i, j].imag,
                        method='linear',
                        bounds_error=False,
                        fill_value=0,
                    ))

                self.rgi_cache[freq] = rgi
                return rgi
