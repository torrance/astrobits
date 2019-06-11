from __future__ import print_function, division

from astropy.coordinates import AltAz, SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import numpy as np


def radec_to_lm(ra, dec, ra0, dec0):
    l = np.cos(dec) * np.sin(ra - ra0)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return np.array([l, m])


def lm_to_radec(l, m, ra0, dec0):
    n = np.sqrt(1 - l**2 - m**2)
    delta_ra = np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    ra = ra0 + delta_ra
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    return np.array([ra, dec])


def radec_to_altaz(ra, dec, time, pos):
    coord = SkyCoord(ra, dec, unit=(u.radian, u.radian))
    coord.time = time + pos.lon.hourangle
    coord = coord.transform_to(AltAz(obstime=time, location=pos))
    return coord.alt.rad, coord.az.rad

def fits_to_radec(hdu):
    wcs = WCS(header=hdu)

    idxs = np.indices(hdu.data.shape)
    # WCS expects indices in header order (ie. reversed)
    idxs = np.flip(idxs, axis=0)
    # Rearrange indices to be of form N, D
    idxs = np.transpose(idxs, axes=[1, 2, 3, 4, 0])
    idxs = np.reshape(idxs, [-1, 4])

    # Calculate world coordinates of grid
    grid_coords = wcs.wcs_pix2world(idxs, 0)  # [(ra, dec, freq, stokes)]
    # ASSUMPTION: fits file is using degrees
    return np.radians(np.reshape(grid_coords[:, 0], hdu.data.shape)), np.radians(np.reshape(grid_coords[:, 1], hdu.shape))



