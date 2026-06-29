"""
Utils functions for ABCD calculations including propagation matrices
"""

import numpy as np


# ABCD matrix - Boyd Table 4.1
def M_free(d):
    return np.array([[1, d],
                     [0, 1]])


def M_focal(f):
    return np.array([[1, 0],
                     [-1/f, 1]])


# Propagation calculations
def stability_condition(A,D):
    """ Stability condition resonators -1 < (A+D)/2 <1 """
    return -1 < (A + D)/2 < 1


def eigen_q(M):
    """Return complex beam parameter q for stable cavity."""
    A, B, C, D = M.ravel()

    if stability_condition(A,D):
        roots = np.roots([C, D - A, -B])
        q = roots[np.imag(roots) > 0][0]
        return q
    else:
        return None


def waist_and_position(q, wavelength):
    """
    Returns:
        w0 : waist size
        z0 : waist position (relative to reference plane)
    """
    if q is None:
        return None, None

    else:
        w0 = np.sqrt(wavelength / (np.pi * np.imag(-1 / q)))
        z0 = -np.real(q)

        return w0, z0
