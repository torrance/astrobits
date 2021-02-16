from astropy.wcs import WCS


def cutoutfits(hdu, center, width):
    hdu = hdu.copy()

    wcs = WCS(hdu.header)
    (x, y, _, _), = wcs.wcs_world2pix([[center.ra.degree, center.dec.degree, 0, 0]], 0)
    x, y = int(x), int(y)
    hdu.header['CRPIX1'] -= max(0, x - width)
    hdu.header['CRPIX2'] -= max(0, y - width)
    hdu.data = hdu.data[:, :, max(0, y-width):y+width, max(0, x-width):x+width]
    hdu.header['NAXIS1'] = hdu.data.shape[3]
    hdu.header['NAXIS2'] = hdu.data.shape[2]

    return hdu


def removeaxes(header):
    header["NAXIS"] = 2
    for kw in ["NAXIS%d", "CRVAL%d", "CRPIX%d", "CDELT%d", "CTYPE%d", "CUNIT%d"]:
        try:
            del header[kw % 3]
            del header[kw % 4]
        except KeyError as e:
            pass

    return header

