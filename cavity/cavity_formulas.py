import numpy as np
from scipy.constants import c


# -- General cavity formulas -- #
def finesse(T, Loss=0.):
    """
    Cavity Finesse for a bow-tie ring cavity.
    :param T: Transmission coefficient
    :param Loss: Intra-cavity Loss
    :return:
    """
    return np.pi * ((1 - T) * (1 - Loss))**(1/4) / (1 - np.sqrt((1 - T) * (1 - Loss)))


def FSR_bowtie(L):
    """
    Free Spectral Range (frequency domain) for a bow-tie ring cavity
    :param L: cavity length
    :return:
    """
    return c / L


def bandwidth(T, Loss, L):
    """
    Calculate bandwidth of bow-tie ring cavity (frequency domain)
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :param L: Cavity length
    :return:
    """
    return FSR_bowtie(L=L) / finesse(T=T, Loss=Loss)
