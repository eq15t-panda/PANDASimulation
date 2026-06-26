import matplotlib.pyplot as plt
import numpy as np


# Finding cavity distances
def cos_theta_function(R, d_curved, l):
    """

    :param R:
    :param d_curved:
    :param l:
    :return:
    """
    return 2 * (R/(d_curved - l))**2 - 1


def finding_unknown_distance(L, R, l, d_curved):
    """
    Calculate missing distances in a ring cavity configuration. Please ensure all lengths are in meters!
    :param L: round-trip length
    :param R: curved mirror radius
    :param l: crystal length
    :param d_curved: distance between curved mirrors, by default None
    :return: the unknown distances as well as the incident angles on the mirrors
    """
    cos_theta = cos_theta_function(R, d_curved, l)
    d_flat = L/(1 + (1/cos_theta)) - d_curved
    OF = d_flat / (2 * cos_theta)
    OC = d_curved / (2 * cos_theta)
    S = (d_curved - l)/2 + OC + OF

    return d_flat, OF, OC, cos_theta, S


def equivalent_cavity(d_flat, d_curved, l, R):
    """

    :param d_flat:
    :param d_curved:
    :param l:
    :param R:
    :return:
    """
    cos_theta = cos_theta_function(R, d_curved, l)
    return d_flat / (2 * cos_theta) + d_curved / (2 * cos_theta) + (d_curved - l) / 2
