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


def FSR_loaded(L, l, n):
    """
    FSR for a loaded cavity with a dispersive medium of index n
    :param L: Cavity length
    :param l: Medium length
    :param n: Medium index
    :return:
    """


def bandwidth(T, Loss, L):
    """
    Calculate bandwidth of bow-tie ring cavity (frequency domain)
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :param L: Cavity length
    :return:
    """
    return FSR_bowtie(L=L) / finesse(T=T, Loss=Loss)

# -- Cavity transfer functions
def transmission(Psi, T, L):
    """
    Complex transmission coefficient.
    """
    return (
        T * np.sqrt(1 - L) * np.exp(1j * Psi / 2)
        / (1 - (1 - T) * (1 - L) * np.exp(1j * Psi))
    )


def reflection(Psi, T, L):
    """
    Complex reflection coefficient.
    """
    return (
        np.sqrt(1 - T)
        * (1 - (1 - L) * np.exp(1j * Psi))
        / (1 - (1 - T) * (1 - L) * np.exp(1j * Psi))
    )


# -- PDH
def pdh_error(Psi, T, L,
              modulation_phase=5.0,
              demod_phase=np.pi/2,
              reflection_detection=True):
    """
    Simulate a PDH error signal.
    """

    if reflection_detection:
        h = reflection
    else:
        h = transmission

    h0 = h(Psi, T, L)
    hp = h(Psi + modulation_phase, T, L)
    hm = h(Psi - modulation_phase, T, L)

    error = -(
        np.real(np.conj(h0) * (hp - hm)) * np.cos(demod_phase)
        - np.imag(np.conj(h0) * (hp + hm)) * np.sin(demod_phase)
    )

    return error
