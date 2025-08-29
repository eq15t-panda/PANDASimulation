#%%

import sys
sys.path.insert(0, '/Users/gadanimatteo/Desktop/SqueezingSimulation')
from cavity.cavity_formulas import z_parameter
from cavity.cavity_formulas import return_waist
from cavity.cavity_formulas import ABCD_Matrix
from cavity.ABCD_matrix import refract_interface
from cavity.ABCD_matrix import thin_lense
from cavity.ABCD_matrix import curved_mirror
from cavity.ABCD_matrix import free_space
from utils.settings import settings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# %%
#Set of parameters

d_curved = 0.185
L = 0.79
R = 0.15
l_crystal = 0.02
f1 = 0.15
f2 = 1
l1 = 0.209
w2 = return_waist(d_curved, L, R, l_crystal)[1]
z_r = np.pi*w2**2/settings.wavelength
w1 = 0.85e-3 
z_rc = np.pi*w1**2/settings.wavelength
q_in = 1j*z_rc
q_out = 1j*z_r


def ABCD_lenses(f1, f2, d12, d2):
    """
    Ray transfer matrix through two lenses

    :param f1: Focal lense of the first lense
    :param f2: Focal lense of the second lense
    :param d12: Distance between the two lenses
    :param d2: Distance between the second lens and the entrance of the cavity
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :return: Tuple (A1, B1, C1, D1)
    """
    M = (
        free_space(d2) @         # L2 -> cavity
        thin_lense(f2) @       # L2
        free_space(d12) @          # L1 -> L2
        thin_lense(f1)             # L1
    )
    A, B, C, D = M.flatten()
    return A, B, C, D

#%%
list_omeg = []
list_d2 = np.linspace(0,0.2,200)
for i in range(len(list_d2)):   
    A,B,C,D = ABCD_lenses(f1,f2,0.11,list_d2[i])
    if i==2 : 
        print(A,B,C,D)
    q_out2 = (A*q_in+B)/(C*q_in+D)
    omega = np.sqrt(-settings.wavelength/np.imag(1/q_out2)/np.pi)*1e6
    list_omeg.append(omega)

fig, ax = plt.subplots()
ax.plot(list_d2, list_omeg)
ax.axhline(242)
