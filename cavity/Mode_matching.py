# %%
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
from scipy.optimize import least_squares
from scipy.optimize import minimize
import matplotlib.colors as colors


# %%
#Set of parameters

d_curved = 0.185
L = 0.79
R = 0.15
l_crystal = 0.02
f1 = 0.15
f2 = 0.2
l1 = 0.209
w2 = return_waist(d_curved, L, R, l_crystal)[1]
z_r = np.pi*w2**2/settings.wavelength
w1 = 0.85e-3 
z_rc = np.pi*w1**2/settings.wavelength
q_in = 1j*z_rc
q_out = 1j*z_r

#%%

# -- Ray propagation -- from fiber to cavity #



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
# %%


'Define the two equations to be solve numerically - we assume that f1 and f2 '
'are fixed'


def systeme(d_curved, L, R, l_crystal, f1, f2, d12, d2, l1):
    w2 = return_waist(d_curved, L, R, l_crystal)[1]
    z_r = np.pi*w2**2/settings.wavelength
    z_rc = 0.85e-3    # waist at the exit of the fiber
    A, B, C, D = ABCD_lenses(f1, f2, d12, d2)
    return np.abs((A*C*z_rc**2 + B*D)/((C**2*z_rc**2+D**2)))**2 + np.abs(z_r - z_rc/(C**2*z_rc**2+D**2))**2


#We only optimize the imaginary part -> beam waist
def systeme2(q_in, q_out, d12, d2):
    A, B, C, D = ABCD_lenses(f1, f2, d12, d2)
    x = (A*q_in + B)/(C*q_in + D)
    return (np.imag(x) - np.imag(q_out))**2 

# %%
# Fonction à minimiser


def residuals(vars):
    d12, d2 = vars
    A, B, C, D = ABCD_lenses(f1, f2, d12, d2)
    eq1 = np.abs((A*C*z_rc**2 + B*D)/((C**2*z_rc**2+D**2)))
    eq2 = np.abs(z_r - z_rc/(C**2*z_rc**2+D**2))
    return eq1**2 + eq2**2

def residuals2(vars):
    d12, d2 = vars
    A, B, C, D = ABCD_lenses(f1, f2, d12, d2)
    x = (A*q_in + B)/(C*q_in + D)
    return (np.imag(x) - np.imag(q_out))**2 

# %%


initial_guess = [0.1, 0.1]
bounds = ((0,100), (0,100))
result = minimize(residuals2, initial_guess, bounds=bounds, method='L-BFGS-B')
# %%


list_d12 = np.linspace(0, 1, 500)
list_d2 = np.linspace(0, 1, 500)
cond = np.zeros((len(list_d12), len(list_d2)))

for i in range(len(list_d12)): 
    for j in range(len(list_d2)):
        #cond[i, j] = systeme2(q_in, q_out, list_d12[i], list_d2[j])
        cond[i, j] = systeme2(q_in, q_out, list_d12[i], list_d2[j])

D12, D2 = np.meshgrid(list_d12, list_d2)

# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

plt.pcolormesh(D12*100, D2*100, cond, norm=colors.LogNorm(vmin=cond.min(), vmax=cond.max()))
plt.ylabel(r'$d_{12}$ (cm)')
plt.xlabel('$d_2$ (cm)')
plt.ylim(0,100)
plt.colorbar()

# %%
np.unravel_index(np.argmin(cond), cond.shape)
# %%
