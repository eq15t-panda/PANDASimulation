import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv  # Bessel functions

import utils.plot_style


# Define parameters
r = 0.95  # field reflectivity
w_ratio = np.linspace(start=0.55, stop=1.5, num=1000)  # omega / FSR

# Define the complex reflection coefficient
exp_term = np.exp(1j * 2 * np.pi * w_ratio)
F = r * (exp_term - 1) / (1 - r**2 * exp_term)

# Calculate reflection intensity
intensity = np.abs(F)**2

# Calculate phase
phase = np.angle(F)
# Split the phase data into positive and negative parts
positive_phase_indices = phase >= 0
negative_phase_indices = phase < 0

# Calculate derivative of reflection intensity
d_intensity = np.gradient(intensity, w_ratio)

# Courtesy of Matteo GADANI
# Parameters of the bow tie cavity
r1 = np.sqrt(88/100)
r2 = np.sqrt(99.98/100)
r3 = r2
r4 = r2

L = 790e-3  # Cavity length
wavelength = 780e-9  # Carrier wavelength
Omega = 20e6  # Modulation frequency
c = 3e8
beta = 0.05          # modulation depth
T = 0.2   # piezo transducer period
Omega_las = 2 * np.pi * c / wavelength
phi = Omega_las * c
phi_side = Omega / c * L

# List for phase and time
list_phi = np.linspace(start=0, stop=3*np.pi, num=1000)
list_time = np.linspace(start=0, stop=T, num=5000)
list_intensity = np.zeros((len(list_phi), len(list_time)))


# Reflexion coefficient of the cavity (bow-tie)
def r_cav(r1, r2, r3, r4, phi):
    """
    Calculate the reflection coefficient of a bow-tie cavity.
    :param r1: Reflection coefficient of the first mirror
    :param r2: Reflection coefficient of the second mirror
    :param r3: Reflection coefficient of the third mirror
    :param r4: Reflection coefficient of the fourth mirror
    :param phi: Phase shift
    :return: Reflection coefficient of the bow-tie cavity
    """
    t1 = np.sqrt(1-r1**2)
    R = r1*r2*r3*r4
    return r1 - t1**2 * R / r1 * np.exp(1j * phi) / (1-R*np.exp(1j*phi))


# Reflexion coefficient of the cavity (Fabry-Perot)
def reflection_amplitude2(r, phi):
    """

    :param r: Reflection coefficient
    :param phi: Phase shift
    :return: Reflection amplitude of a Fabry-Perot cavity
    """
    return r * (np.exp(1j * phi) - 1)/(1 - r**2 * np.exp(1j * phi))


# Cavity length variation with time
def phase(t, Omega, n=6):
    """
    Calculate the phase shift of the cavity as a function of time.
    :param t: Time in seconds
    :param Omega: Modulation frequency in radians per second
    :param n: Number of periods for the modulation (number of peak of resonance for a rise)
    :return: Phase shift in radians
    """
    # # Bring t into a period [0, T]
    t_mod = np.mod(t, T)

    # # Rising ramp up to T/2, then descending ramp
    if t_mod <= T / 2:
        t_eff = t_mod
    else:
        t_eff = T - t_mod  # symmetric descent

    return (Omega / c) * (L + n * wavelength / T * t_eff)


n = 6  # Number of periods for the modulation (number of peak of resonance for a rise)
t_vals = np.linspace(0, 5*2*np.pi/Omega, 10000)
y_vals = [phase(t, Omega, n) for t in t_vals]

# plt.plot(t_vals, y_vals)


# Reflected signal with modulation (J0 for carrier, J1 for sidebands)
def E_refl(r1, r2, r3, r4, beta, Omega, t):
    phi = phase(t, Omega_las)
    phi_side = phase(t, Omega)

    # Reflected fields
    r0 = r_cav(r1, r2, r3, r4, phi)                     # carrier
    r_plus = r_cav(r1, r2, r3, r4, phi + phi_side)      # sideband +
    r_minus = r_cav(r1, r2, r3, r4, phi - phi_side)     # sideband -

    # Modulation with Bessel functions
    E0 = jv(0, beta) * r0
    E1_plus = jv(1, beta) * r_plus
    E1_minus = jv(1, beta) * r_minus

    # Detected signal (intensity)
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


# Calculate the reflected signal for each time point
signal = [E_refl(r1, r2, r3, r4, beta, Omega, t)[3] for t in list_time]

# Calculate the period
period = T / n

# Plot
plt.figure(figsize=(9, 4))
plt.plot(list_time, signal, 'k', lw=1.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel(r"Time (s)")
plt.ylabel("Reflected intensity AC")
plt.title("Reflected intensity against time")
plt.xlim(0, period)  # Restrict x-axis to one period starting from the first peak
plt.grid(True)
plt.tight_layout()
plt.show()


r0 = r_cav(r1, r2, r3, r4, phi)                     # carrier
r_plus = r_cav(r1, r2, r3, r4, phi + phi_side)      # sideband +
r_minus = r_cav(r1, r2, r3, r4, phi - phi_side)     # sideband -


E0 = jv(0, beta) * r0
E1_plus = jv(1, beta) * r_plus
E1_minus = jv(1, beta) * r_minus

DC = np.abs(E0)**2 + np.abs(E1_minus)**2 + np.abs(E1_plus)**2

diff_r = r0 * np.conj(r_plus) - np.conj(r0) * r_minus

AC_r = 2 * jv(1, beta) * jv(0, beta) * np.real(diff_r)
AC_i = 2 * jv(1, beta) * jv(0, beta) * np.imag(diff_r)


def calcul_fft(signal, fe):
    """
    Calculates and displays the FFT of a time-domain signal.

    signal : list or numpy array
        Time-domain signal (amplitude as a function of time)
    fe : float
        Sampling frequency in Hz (number of points per second)
    """
    # Conversion en array numpy (au cas où c'est une liste)
    signal = np.array(signal)

    N = len(signal)  # Number of points
    fft_vals = np.fft.fft(signal)  # FFT
    fft_freq = np.fft.fftfreq(N, 1 / fe)  # Frequency axes

    # Only keep positive part
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    fft_magnitude = np.abs(fft_vals[pos_mask]) / N  # Normalisation

    # Display
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, fft_magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency spectrum of the signal")
    plt.grid(True)
    plt.show()

    return freqs, fft_magnitude


plt.plot(list_time, [np.abs(r_cav(r1, r2, r3, r4, phase(t, Omega_las)))**2 for t in list_time], 'k')
plt.xlabel('Time (s)')
plt.ylabel(r'$|r(\omega)|^2$')
plt.title('Bow tie cavity reflection coefficient')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.ylim(0.98, 1.002)
# plt.legend()
plt.tight_layout()
# plt.show()

signal = [np.imag(E_refl(r1, r2, r3, r4, beta, Omega, t)[5]) for t in list_time]
# calcul_fft(signal, fe=10 * Omega)
