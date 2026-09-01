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
def r_c(Psi, T, L):
    return np.sqrt(1 - T) * (1 - (1 - L) * np.exp(1j * Psi)) / \
           (1 - (1 - T) * (1 - L) * np.exp(1j * Psi))

def error_signal(f, Omega_m, T, L, FSR, phi, beta=0.1):
    Psi0 = 2 * np.pi * f / FSR
    Psim = 2 * np.pi * Omega_m / FSR
    h0 = r_c(Psi0, T, L)
    hp = r_c(Psi0 + Psim, T, L)
    hm = r_c(Psi0 - Psim, T, L)
    P0, Psb = 1.0, beta**2 / 4
    return -np.sqrt(P0 * Psb) * (
        (np.conj(h0) * (hp - hm)).real * np.cos(phi)
        - (np.conj(h0) * (hp + hm)).imag * np.sin(phi)
    )
