import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

"""
Based on 
DOI: 10.1080/09500340.2018.1437228 
'Joint spectral characterization of photon-pair sources'
Kevin Zielnicki et al
"""

# ----- PARAMETERS -----
n_points = 1000
nu_max = 0.05  # GHz range around degeneracy (narrow for cavity-limited bandwidth); calculate the JSA within ±50 MHz around the degenerate frequency.

# Pump bandwidth for CW laser in cavity:
# The CW pump is nearly monochromatic, but the cavity imposes its own spectral filtering.
# Cavity bandwidth Δ = 6 MHz → ~0.006 GHz. This acts as an additional filter on the signal/idler frequencies.
pump_bw = 0.006  # GHz effective cavity-limited bandwidth

L = 20e-3         # Crystal length (m)
lambda0 = 780e-9  # Central wavelength (m)
c = 3e8

# ----- GRID -----
nu_s = np.linspace(start=-nu_max, stop=nu_max, num=n_points)
nu_i = np.linspace(start=-nu_max, stop=nu_max, num=n_points)
NS, NI = np.meshgrid(nu_s, nu_i)

# ----- PUMP ENVELOPE (Gaussian, narrow for CW + cavity filter) -----
A = np.exp(- (NS + NI)**2 / (2*pump_bw**2))

# ----- PHASE MATCHING (sinc approximated) -----
# For narrowband cavity, phase-matching is dominated by cavity filtering, so we can still approximate.
phi = np.sinc((NS - NI) * L)

# ----- CAVITY FILTERING -----
# Apply Lorentzian cavity transmission for both signal and idler.
Delta = 0.006  # GHz half-width at half maximum
cavity_filter = 1 / (1 + (NS/Delta)**2) * 1 / (1 + (NI/Delta)**2)

# ----- JSA -----
JSA = A * phi * cavity_filter
JSA /= np.linalg.norm(JSA)

# ----- SCHMIDT DECOMPOSITION -----
U, S, Vh = svd(JSA)
lambdas = S**2 / np.sum(S**2)
K = 1 / np.sum(lambdas**2)

threshold = 1e-3
significant_idx = lambdas > threshold
lambdas_cut = lambdas[significant_idx]

print(f"Schmidt number K ≈ {K:.2f}")
print(f"Number of significant modes: {len(lambdas_cut)}")

# ----- PLOTS -----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# JSA Intensity
im0 = ax[0].imshow(np.abs(JSA)**2, extent=[nu_s[0], nu_s[-1], nu_i[0], nu_i[-1]], origin='lower')
ax[0].set_xlabel("Signal detuning (GHz)")  # Detuning is the frequency offset of signal (nu_s) or idler (nu_i) from their central degenerate frequency.
ax[0].set_ylabel("Idler detuning (GHz)")
ax[0].set_title("JSA with Cavity Bandwidth Δ=6 MHz")
fig.colorbar(im0, ax=ax[0])

# Schmidt coefficient histogram
ax[1].bar(range(len(lambdas_cut)), lambdas_cut)
ax[1].set_title(f"Schmidt Coefficients (K={K:.2f})")
ax[1].set_xlabel("Mode index")
ax[1].set_ylabel("Weight")

plt.tight_layout()
plt.show()
