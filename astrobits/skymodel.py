from __future__ import print_function, division

import sys

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def parse(f):
    models = []

    next(f)  # Skip version info
    for line in f:
        parts = line.split()
        if parts[0] == 'source':
            models.append(source_parser(f))
        elif parts[0] == '#':
            pass
        else:
            raise SkyModelParseError("Unexpected line: %s" % line)

    return models


def source_parser(f):
    name, components = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'name':
            name = ' '.join(parts[1:]).strip('"')
        elif parts[0] == 'component':
            components.append(component_parser(f))
        elif parts[0] == '}':
            if name and components:
                return Model(name, components)
            else:
                raise SkyModelParseError("Unexpected }")
        else:
            raise SkyModelParseError("Skymodel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


def component_parser(f):
    position, sed, measurements = None, None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'position':
            position = SkyCoord(parts[1], parts[2], unit=(u.hourangle, u.degree))
        elif parts[0] == 'type':
            pass
        elif parts[0] == 'shape':
            pass
        elif parts[0] == 'measurement':
            measurements.append(measurement_parser(f))
        elif parts[0] == 'sed':
            sed = sed_parser(f)
        elif parts[0] == '}':
            if measurements and sed:
                raise SkyModelParseError("Don't know how to handle sed and measurments together")
            elif position and measurements:
                return MeasurementComponent(position, measurements)
            elif position and sed:
                return SEDComponent(position, sed)
            else:
                raise SkyModelParseError("Unexpected }")
        else:
            raise SkyModelParseError("Skymodel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


def measurement_parser(f):
    frequency, fluxdensity = None, []

    for line in f:
        parts = line.split()
        if parts[0] == 'frequency':
            # Assume MHz for now
            frequency = float(parts[1]) * 1E6
        elif parts[0] == 'fluxdensity':
            fluxdensity = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
        elif parts[0] == '}':
            if frequency and fluxdensity:
                return [frequency] + fluxdensity
            else:
                raise SkyModelParseError("Unexpected {")
        else:
            raise SkyModelParseError("SkyModel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


def sed_parser(f):
    freq, flux, alpha1, alpha2 = None, None, None, None

    for line in f:
        parts = line.split()
        if parts[0] == 'frequency':
            # Assume MHz for now
            freq = float(parts[1]) * 1E6
        elif parts[0] == 'fluxdensity':
            # Only consider stokes I
            flux = float(parts[2])
        elif parts[0] == 'spectral-index':
            # Assume maximum of 2 alpha terms
            alpha1 = float(parts[2])
            alpha2 = float(parts[3]) if parts[3] != '}' else 0
        elif parts[0] == '}':
            sed = [freq, flux, alpha1, alpha2]
            if None not in sed:
                return sed
            else:
                raise SkyModelParseError("Incomplete sed")
        else:
            raise SkyModelParseError("SkyModel parsing error: %s" % line)

    raise SkyModelParseError("Unexpected EOF")


def writeto(f, models):
    print("skymodel fileformat 1.1", file=f)
    for model in models:
        print(str(model), file=f)


class Model(object):
    def __init__(self, name, components):
        self.name = name
        self.components = components

    def __str__(self):
        model = "source {\n"
        model += "  name \"%s\"\n" % self.name
        for comp in self.components:
            model += str(comp) + "\n"
        model += "}"
        return model

    def angular_mean(self):
        """
        Find a ra, dec value that is the mean of component positions.
        For a single component source, this will just be the first component's
        position.
        """
        ras = np.array([comp.ra for comp in self.components])
        decs = np.array([comp.dec for comp in self.components])

        # Covert coordinates into spherical coordinates
        # (theta = polar angle, phi = azimuthal angle)
        thetas = (decs - np.pi / 2) * -1
        phis = ras

        # Convert to cartesian and average
        x = np.mean(np.sin(thetas) * np.cos(phis))
        y = np.mean(np.sin(thetas) * np.sin(phis))
        z = np.mean(np.cos(thetas))

        # Convert back to spherical
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        # Convert from spherical to ra, dec
        ra = phi % (2 * np.pi)
        dec = theta * -1 + np.pi / 2

        return ra, dec

    @property
    def ra(self):
        return self.angular_mean()[0]

    @property
    def dec(self):
        return self.angular_mean()[1]

    def apparent(self, midfreq):
        return sum([sum(comp.apparent(midfreq)) for comp in self.components])


class Component(object):
    def __init__(self, position):
        self.position = position
        self.apparent_cache = {}

    @property
    def ra(self):
        return self.position.ra.rad

    @property
    def dec(self):
        return self.position.dec.rad

    def flux(self, frequency):
        """
        Returns stokes I flux value for given frequency
        """
        logfreq = np.log(frequency)
        logflux = 0

        for i, c in enumerate(reversed(self.coeffs)):
           logflux += c * logfreq**i

        return np.exp(logflux)

    def apparent(self, frequency):
        """
        Returns XX, YY flux values for a given frequency
        """
        frequency = np.asscalar(np.array(frequency))

        try:
            return self.apparent_cache[frequency]
        except KeyError:
            jones = self.beam.jones(self.ra, self.dec, frequency)[0]
            flux = self.flux(frequency)

            # apparent = jones x linear x jones^H (where H is the Hermitian transponse)
            # We are assuming XY and YX are neglible, ie.:
            # apparentXX = J0 * J0* * XX + J1 * J1* * YY
            # apparentYY = J2 * J2* * XX + J3 * J3* * YY
            apparent_xx = abs(jones[0, 0])**2 * flux + abs(jones[0, 1])**2 * flux
            apparent_yy = abs(jones[1, 0])**2 * flux + abs(jones[1, 1])**2 * flux

            self.apparent_cache[frequency] = (apparent_xx, apparent_yy)
            return (apparent_xx, apparent_yy)


class SEDComponent(Component):
    def __init__(self, position, sed):
        """
        sed = [reference frequency, flux at ref. freq., alpha1, alpha2 (ie. curvature)]
        """
        super(SEDComponent, self).__init__(position)

        self.sed = sed
        freq, flux, alpha1, alpha2 = sed
        const = np.log(flux) - alpha1 * np.log(freq) - alpha2 * np.log(freq)**2
        self.coeffs = np.array([0, alpha2, alpha1, const])

    def __str__(self):
        comp = "  component {\n"
        comp += "    type point\n"
        comp += "    position %s\n" % self.position.to_string("hmsdms")
        comp += "    sed {\n"
        comp += "      frequency %f MHz\n" % (self.sed[0] / 1e6)
        comp += "      fluxdensity Jy %f 0 0 0\n" % self.sed[1]
        comp += "      spectral-index { %f %f }\n" % (self.sed[2], self.sed[3])
        comp += "    }\n"
        comp += "  }"
        return comp


class MeasurementComponent(Component):
    def __init__(self, position, measurements):
        """
        measurements = [[freq, flux]]
        """
        super(MeasurementComponent, self).__init__(position)

        self.measurements = np.array(measurements)
        logfreq = np.log(self.measurements.T[0])
        logflux = np.log(self.measurements.T[1])
        self.coeffs = np.polyfit(logfreq, logflux, min(len(logfreq) - 1, 3))

    def __str__(self):
        comp = "  component {\n"
        comp += "    type point\n"
        comp += "    position %s\n" % self.position.to_string("hmsdms")
        for measurement in self.measurements:
            comp += "    measurement {\n"
            comp += "      frequency %f MHz\n" % (measurement[0] / 1e6)
            comp += "      fluxdensity Jy %f %f %f %f\n" % (measurement[1], measurement[2], measurement[3], measurement[4])
            comp += "    }\n"
        comp += "  }"
        return comp

class SkyModelParseError(Exception):
    pass

