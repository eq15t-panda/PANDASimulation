# %%


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # fonctions de Bessel

# %%


# Parameters of the bow tie cavity
r1 = np.sqrt(88/100)
r2 = np.sqrt(99.98/100)
r3 = r2
r4 = r2
L = 790e-3  # taille de la cavité
lamb = 852e-9  # longueur d'onde du laser
Omega = 20e8  # fréquence de modulation
c = 3e8
beta = 0.05          # indice de modulation
T = 0.2   # période du piezo 
Omega_las = 2*np.pi*c/lamb


# %%


# Liste phase and time
list_phi = np.linspace(0, 3*np.pi, 1000) 
list_time = np.linspace(0, T, 5000)
list_intensity = np.zeros((len(list_phi), len(list_time)))

# %%


# Coefficient de réflexion bow tie
def r_cav(r1, r2, r3, r4, phi):
    """_summary_

    Args:
        r1 (_type_): _description_
        r2 (_type_): _description_
        r3 (_type_): _description_
        r4 (_type_): _description_
        phi (_type_): _description_

    Returns:
        _type_: _description_
    """
    t1 = np.sqrt(1-r1**2)
    R = r1*r2*r3*r4
    return r1 - t1**2*R/r1*np.exp(1j*phi)/(1-R*np.exp(1j*phi))


# Coefficient de réflexion Fabry Perot
def reflection_amplitude2(r, phi):
    return r*(np.exp(1j*phi)-1)/(1-r**2*np.exp(1j*phi))


plt.plot(list_phi, np.abs(r_cav(r1, r2, r3, r4, list_phi))**2)

# %%


# Variation de L avec le temps
def phase(t, Omega):
    # Ramener t dans une période [0, T]
    t_mod = np.mod(t, T)

    # Rampe montante jusqu’à T/2, descendante ensuite
    if t_mod <= T / 2:
        t_eff = t_mod
    else:
        t_eff = T - t_mod  # redescente symétrique

    return (Omega / c) * (L + 8 * lamb / T * t_eff)


t_vals = np.linspace(0, 5*T, 10000)
y_vals = [phase(t, Omega) for t in t_vals]

plt.plot(t_vals, [np.abs(r_cav(r1, r2, r3, r4, phase(t,Omega_las)))**2 for t in t_vals])


# %%

# %%


# Signal réfléchi avec modulation (J0 pour porteuse, J1 pour sidebands)
def E_refl(r1, r2, r3, r4, beta, Omega, t):
    Omega_las = 2*np.pi*c/lamb
    phi = phase(t, Omega_las)
    phi_side = phase(t, Omega)

    # Champs réfléchis
    r0 = r_cav(r1, r2, r3, r4, phi)        # porteuse
    r_plus = r_cav(r1, r2, r3, r4, phi + phi_side)    # sideband +
    r_minus = r_cav(r1, r2, r3, r4, phi - phi_side)   # sideband -

    # Champs modulés
    E0 = jv(0, beta) * r0
    E1_plus = jv(1, beta) * r_plus
    E1_minus = jv(1, beta) * r_minus

    # Signal détecté (intensité sur photodiode)
    E_total = E0 + E1_plus * np.exp(1j * Omega * t) - E1_minus * np.exp(-1j * Omega * t)
    Int = np.abs(E_total)**2

    # Signal DC 
    DC = np.abs(E0)**2 + np.abs(E1_minus)**2 + np.abs(E1_plus)**2

    # Signal AC
    diff_r = r0*np.conjugate(r_plus) - np.conjugate(r0)*r_minus
    AC_r = 2*jv(1, beta)*jv(0, beta)*np.real(diff_r)
    AC_i = 2*jv(1, beta)*jv(0, beta)*np.imag(diff_r)
    AC_total = AC_r*np.cos(Omega*t) + AC_i*np.sin(Omega*t)

    return Int, DC, AC_r, AC_i, AC_total, diff_r
# %%


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})


# Tracé 
plt.figure(figsize=(9, 4))
plt.plot(list_time, [E_refl(r1, r2, r3, r4, beta, Omega, t)[3] for t in list_time], 'k', lw=1.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel(r"Time (s)", fontsize = 23) 
plt.ylabel("Reflection coefficient", fontsize = 23)
#plt.title("Intensité réfléchie AC")
#plt.ylim(0.98, 1.003)
plt.xlim(0.0, 0.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# %%

phi = 2*np.pi*L/lamb
phi_side = Omega/c*L
r0 = r_cav(r1, r2, r3, r4, phi)        # porteuse
r_plus = r_cav(r1, r2, r3, r4, phi + phi_side)    # sideband +
r_minus = r_cav(r1, r2, r3, r4, phi - phi_side)   # sideband -

#r0 = reflection_amplitude2(0.99, phi)        # porteuse
#r_plus = reflection_amplitude2(0.99, phi + phi_side)    # sideband +
#r_minus = reflection_amplitude2(0.99, phi - phi_side)   # sideband -


E0 = jv(0, beta) * r0
E1_plus = jv(1, beta) * r_plus
E1_minus = jv(1, beta) * r_minus

DC = np.abs(E0)**2 + np.abs(E1_minus)**2 + np.abs(E1_plus)**2

diff_r = r0*np.conj(r_plus) - np.conj(r0)*r_minus


AC_r = 2*jv(1,beta)*jv(0,beta)*np.real(diff_r)
AC_i = 2*jv(1,beta)*jv(0,beta)*np.imag(diff_r)

# %%


def calcul_fft(signal, fe):
    """
    Calcule et affiche la FFT d'un signal temporel.
    
    signal : list or numpy array
        Signal temporel (amplitude en fonction du temps)
    fe : float
        Fréquence d'échantillonnage en Hz (nombre de points par seconde)
    """
    # Conversion en array numpy (au cas où c'est une liste)
    signal = np.array(signal)

    N = len(signal)                 # Nombre de points
    fft_vals = np.fft.fft(signal)  # FFT brute
    fft_freq = np.fft.fftfreq(N, 1/fe)  # Axes de fréquences

    # On ne garde que la partie positive (utile)
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    fft_magnitude = np.abs(fft_vals[pos_mask]) / N  # Normalisation

    # Affichage
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_magnitude)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Spectre en fréquence du signal")
    plt.grid(True)
    plt.show()

    return freqs, fft_magnitude


signal = [np.imag(E_refl(r1, r2, r3, r4, beta, Omega, t)[5]) for t in list_time]
calcul_fft(signal, fe=10*Omega)


# %%



plt.plot(list_time, [np.abs(r_cav(r1, r2, r3, r4, phase(t, Omega_las)))**2 for t in list_time], 'k')
plt.xlabel('Time (s)')
plt.ylabel(r'$|r(\omega)|^2$')
plt.title('Bow tie cavity reflection coefficient')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.ylim(0.98, 1.002)
plt.legend()
plt.tight_layout()
plt.show()
# %%
