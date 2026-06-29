import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from utils.settings import settings

# Constants
alpha1 = 50
alpha2 = 2 * alpha1


def compute_K(alpha1, alpha2, xi, L):
    alpha = alpha1 + 0.5 * alpha2
    K = alpha * L / (2 * xi)
    return K


def compute_F_fast(sigma, xi, K, N=400):
    tau = np.linspace(-xi, xi, N)
    tau_prime = np.linspace(-xi, xi, N)
    d_tau = tau[1] - tau[0]

    T, T_prime = np.meshgrid(tau, tau_prime)
    exponent = - K * (T + T_prime) + 1j * sigma * (T - T_prime)
    denominator = (1 + 1j * T) * (1 - 1j * T_prime)

    integrand = np.exp(exponent) / denominator
    integral = np.sum(integrand) * d_tau ** 2

    return np.real(integral) / (4 * np.pi ** 2)


def compute_h_fast(sigma, xi, L, alpha1=alpha1, alpha2=alpha2):
    K = compute_K(alpha1, alpha2, xi, L)
    F_val = compute_F_fast(sigma, xi, K)
    # print(f"K = {K:.4e} for L = {L:.3e}")
    return (np.pi ** 2 / xi) * F_val


def optimize_hm_fast(xi, L, alpha1=alpha1, alpha2=alpha2):
    objective = lambda sigma: -compute_h_fast(sigma, xi, L, alpha1, alpha2)
    res = minimize_scalar(objective, bounds=(-10, 10), method='bounded')
    return -res.fun, res.x


def WaistFromXi(xi, L, wavelength, index=1):
    b = L / xi
    return np.sqrt((b * wavelength) / (2 * np.pi))


if __name__ == "__main__":
    # Generate curves for different crystal lengths
    crystal_lengths = [1e-3, 20e-3]
    xi_vals = np.logspace(-1, 1, 70)
    results = {}

    for L in crystal_lengths:
        print(f'doing L={L}')
        h_vals = []
        for xi in xi_vals:
            h, _ = optimize_hm_fast(xi, L)
            h_vals.append(h)
        h_vals = np.array(h_vals)
        max_idx = np.argmax(h_vals)
        xi_opt = xi_vals[max_idx]
        waist_opt = WaistFromXi(xi_opt, L, settings.wavelength)
        h_max = h_vals[max_idx]
        results[f"L={L*1e3:.0f}mm"] = {
            "xi_vals": xi_vals,
            "h_vals": h_vals,
            "xi_opt": xi_opt,
            "waist_opt": waist_opt,
            "h_max": h_max
        }

    # K = 0 reference (independent of L)
    h_vals_0 = []
    for xi in xi_vals:
        h_0, _ = optimize_hm_fast(xi, L=1, alpha1=0, alpha2=0)  # L arbitrary since K=0
        h_vals_0.append(h_0)
    h_vals_0 = np.array(h_vals_0)
    max_idx_0 = np.argmax(h_vals_0)
    xi_opt_0 = xi_vals[max_idx_0]
    waist_opt_0 = WaistFromXi(xi_opt_0, L=20e-3, wavelength=settings.wavelength)

    h_max_0 = h_vals_0[max_idx_0]
    results["K=0"] = {
        "xi_vals": xi_vals,
        "h_vals": h_vals_0,
        "xi_opt": xi_opt_0,
        "waist_opt": waist_opt_0,
        "h_max": h_max_0
    }

    # Plotting
    plt.figure(figsize=(10, 6))

    for label, data in results.items():
        display_label = (rf"${label}$: "
                         rf"$\xi_{{\mathrm{{opt}}}}={data['xi_opt']:.2f},\, "
                         rf"w_0={data['waist_opt'] * 1e6:.1f}\,\mu m,\, "
                         rf"h_{{\max}}={data['h_max']:.3f}$")
        plt.plot(data["xi_vals"], data["h_vals"], label=display_label)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\xi$", fontsize=14)
    plt.ylabel(r"$h_m(\xi)$", fontsize=14)
    plt.title("Optimized $h_m(\\xi)$ with Absorption (for $K=0$, $L=20$mm)", fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
