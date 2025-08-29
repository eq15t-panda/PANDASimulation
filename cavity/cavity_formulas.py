#%%
import numpy as np
import sys
sys.path.insert(0,'/Users/gadanimatteo/Documents/Stage LKB/SqueezingSimulation')
import cavity.finding_distances as fd
from utils.settings import settings
from cavity.ABCD_matrix import free_space
from cavity.ABCD_matrix import curved_mirror
from cavity.ABCD_matrix import thin_lense
from cavity.ABCD_matrix import refract_interface



# -- General cavity formulas -- #
def Finesse(T, Loss):
    """
    Cavity Finesse
    :param T: Transmission coefficient
    :param Loss: Intra-cavity Loss
    :return:
    """
    return np.pi * (((1 - T) * (1 - Loss)) ** (1/4)) / (1 - np.sqrt((1 - T) * (1 - Loss)))


def FSR(L):
    """
    Free Spectral Range (frequency domain)
    :param L: Cavity length
    :return:
    """
    return settings.c / L


def Bandwidth_bowtie(T, Loss, L):
    """
    Calculate bandwidth of bow-tie ring cavity (frequency domain)
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :param L: Cavity length
    :return:
    """
    return FSR(L=L) / Finesse(T=T, Loss=Loss)


def Bandwidth_linear(cavity_length, transmission_coefficient):
    """
    Calculates the bandwidth of a linear cavity from a couple (L, T)
    :param cavity_length: in meter
    :param transmission_coefficient:
    :return:
    """
    # return (c / (2 * cavity_length)) * (transmission_coefficient / (2 * np.pi))
    return (3e8 * transmission_coefficient) / (4 * np.pi * cavity_length)


def Escape_efficiency(T, Loss):
    """
    
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :return: 
    """
    return T / (T + Loss)


def Pump_threshold(T, Loss, E):
    """
    Pump threshold power
    :param T: Transmission coefficient
    :param Loss: Loss: Intra-cavity loss
    :param E: Effective non-linearity of optical medium
    :return:
    """
    return ((T + Loss) ** 2) / (4 * E)


# -- Ray propagation -- from spheric to plan #
def ABCD_Matrix(L, d_curved, R, l_crystal, index_crystal=settings.crystal_index):
    """
    Ray transfer matrix for a half single pass in a ring bow-tie cavity.

    :param L: Cavity round-trip length
    :param d_curved: Distance between curved mirrors
    :param R: Radii of curvature of curved mirror
    :param l_crystal: Length of non-linear crystal
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :return: Tuple (A1, B1, C1, D1)
    """
    M = (
        free_space((L - d_curved)/2) @         # M1 → M2
        curved_mirror(R) @       # M2
        free_space((d_curved - l_crystal)/2) @          # M2 → M3
        refract_interface(index_crystal, 1) @       # M3
        free_space(l_crystal/2) 
    )
    A1, B1, C1, D1 = M.flatten()
    

    # A1 = 1 - (L - d_curved) / R
    # B1 = ((L - d_curved) / 2 + ((d_curved - l_crystal) / 2) * (1 - (L - d_curved) / R)) / index_crystal + l_crystal * (1 - (L - d_curved) / R) / 2
    # C1 = - 2 / R
    # D1 = (1 - (d_curved - l_crystal) / R) / index_crystal - l_crystal / R

    return A1, B1, C1, D1

    # MASADA
    # E = -l_crystal/2 + (d_curved - l_crystal)/2 * index_crystal
    # F = -2/R * E + index_crystal
    #
    # A = 1 - (L - d_curved) / R
    # B = E + (L - d_curved)/2 * F
    # C = -2/R
    # D = F
    # return A, B, C, D


# -- Tamagawa / Svelto -- #
def z_parameter(A1, B1, C1, D1):
    """
    Calculates z parameter
    :param A1:
    :param B1:
    :param C1:
    :param D1:
    :return:
    """
    z1 = - (B1 * D1) / (A1 * C1)  # wavefront at mirror 1
    z2 = - (A1 * B1) / (C1 * D1)  # wavefront at mirror 2

    return z1, z2


def return_waist(d_curved, L, R, l_crystal, index_crystal=settings.crystal_index, wavelength=settings.wavelength): 
    A1, B1, C1, D1 = ABCD_Matrix(L=L, d_curved=d_curved, R=R, l_crystal=l_crystal, index_crystal=index_crystal)
    z1, z2 = z_parameter(A1, B1, C1, D1)
    w1 = np.sqrt((wavelength / (index_crystal * np.pi)) * z1**(1/2))
    w2 = np.sqrt((wavelength / (np.pi)) * z2**(1/2))
    return w1, w2


def Beam_waist(d_curved, L, R, l_crystal, index_crystal=settings.crystal_index, wavelength=settings.wavelength):
    d_curved = np.atleast_1d(d_curved)
    scalar_input = d_curved.ndim == 0 or len(d_curved) == 1

    z1_list, z2_list, w1_list, w2_list = [], [], [], []
    valid_1, valid_2 = [], []

    for dc in d_curved:
        A1, B1, C1, D1 = ABCD_Matrix(L=L, d_curved=dc, R=R, l_crystal=l_crystal, index_crystal=index_crystal)
        z1, z2 = z_parameter(A1, B1, C1, D1)

        z1_list.append(z1)
        z2_list.append(z2)

        if z1 >= 0:
            temp1 = np.sqrt(z1)
            w1 = np.sqrt((wavelength / (index_crystal * np.pi)) * temp1)
            valid_1.append(True)
        else:
            w1 = np.nan
            valid_1.append(False)

        if z2 >= 0:
            temp2 = np.sqrt(z2)
            w2 = np.sqrt((wavelength / np.pi) * temp2)
            valid_2.append(True)
        else:
            w2 = np.nan
            valid_2.append(False)

        w1_list.append(w1)
        w2_list.append(w2)

    z1_array = np.array(z1_list)
    z2_array = np.array(z2_list)
    w1_array = np.array(w1_list)
    w2_array = np.array(w2_list)

    valid_indices_1 = np.where(np.array(valid_1))[0]
    valid_indices_2 = np.where(np.array(valid_2))[0]

    # Si input scalaire → retourner des scalaires
    if scalar_input:
        return z1_array[0], z2_array[0], w1_array[0], w2_array[0], (valid_indices_1, valid_indices_2)
    else:
        return z1_array, z2_array, w1_array, w2_array, (valid_indices_1, valid_indices_2)



# -- Kaertner classnotes -- #
def waist_mirror1(R1, R2, L, wavelength):
    return (((wavelength * R1) / np.pi) ** 2 * ((R2 - L) / (R1 - L)) * (L / (R1 + R2 - L))) ** (1 / 4)


def waist_mirror3(R1, R2, L, wavelength):
    return (((wavelength * R2) / np.pi) ** 2 * ((R1 - L) / (R2 - L)) * (L / (R1 + R2 - L))) ** (1 / 4)


def waist_intracavity(R1, R2, L, wavelength):
    return ((wavelength / np.pi) ** 2 * (L * (R1 - L) * (R2 - L) * (R1 + R2 - L) / ((R1 + R2 - 2 * L) ** 2))) ** (1 / 4)


# -- Laurat / Boyd -- #
def effective_length(L, l, refractive_index):
    """
    Calculates the effective length of a cavity with a non-linear crystal of length l and index n
    :param L:
    :param l:
    :param refractive_index:
    :return:
    """
    return L - l * (1 - 1/refractive_index)


def effective_Rayleigh_length(L, l, refractive_index, R):
    """
    New Rayleigh length for OPO
    :param L:
    :param l:
    :param refractive_index:
    :param R:
    :return:
    """
    eff_length = effective_length(L, l, refractive_index)
    return np.sqrt(eff_length * (R - eff_length))


def stability_condition(A, D):
    """
    Compute the stability condition for a symmetrical cavity
    :param A:
    :param D:
    :return:
    """
    s = (A+D) / 2
    # check where -1 < (A+D)/2 < 1
    return np.logical_and(s > -1, s < 1)

# %%
