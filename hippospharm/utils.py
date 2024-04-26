import numpy as np


def transform_cartesian_to_spherical(x, y,z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.where(r == 0, 0.0001, r)
    theta = np.arccos(z * 1. / r)
    phi = np.arctan2(y, x) + np.pi
    return r, theta, phi


def transform_spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z