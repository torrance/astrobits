import numpy as np


def radec_to_cartesian(ra, dec):
    theta = np.pi / 2 - dec
    phi = ra

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def cartesian_to_radec(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    ra = phi
    dec = np.pi / 2 - theta

    return ra, dec


def rotate_about_x(x, y, z, angle):
    matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])
    return np.matmul(matrix, [x, y, z])


def rotate_about_y(x, y, z, angle):
    matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    return np.matmul(matrix, [x, y, z])


def rotate_about_z(x, y, z, angle):
    matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    return np.matmul(matrix, [x, y, z])
