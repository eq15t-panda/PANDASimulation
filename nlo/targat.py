import numpy as np
from scipy.optimize import brentq, minimize_scalar


def xi_from_waist(w0, l, wavelength, n):
    """
    Convert beam waist w0 (m) to BK focusing parameter xi using
    xi = l * lambda / (2 * n * pi * w0^2)
    """
    return (l * wavelength) / (2.0 * n * np.pi * (w0 ** 2))


def compute_h(sigma, xi, alpha, N=500):
    """Numerically compute h(sigma, xi, alpha) using a 2D trapezoidal rule.

    Parameters
    - sigma: scalar
    - xi: scalar > 0
    - alpha: scalar (can be zero)
    - N: grid points per dimension (increasing improves accuracy, slows down)

    Returns real(h)
    """
    # integration domain
    tau = np.linspace(-xi, xi, N)
    # create 2D grid: rows -> tau_i, cols -> tau_j (tau')
    T, Tp = np.meshgrid(tau, tau, indexing='ij')

    # integrand
    numer = np.exp(1j * sigma * (T - Tp)) * np.exp(-alpha * (T + Tp))
    denom = (1 + 1j * T) * (1 - 1j * Tp)
    integrand = numer / denom

    # inner integral over tau' (axis=1), then outer over tau (axis=0)
    inner = np.trapezoid(integrand, x=tau, axis=1)
    total = np.trapezoid(inner, x=tau)

    h = total / (4.0 * xi)
    return np.real(h)


def find_sigma_opt(xi, alpha, sigma_bounds=(0.0, 5.0), N_coarse=200, N_refine=500):
    """Find sigma that maximizes h for given xi and alpha.
    Do a two-stage search: coarse bounded minimization (neg h) then refine around found minimum.
    Returns (sigma_opt, h_max)
    """
    # objective for minimizer (minimize negative h)
    def obj(s):
        return -compute_h(s, xi, alpha, N=N_coarse)

    res = minimize_scalar(obj, bounds=sigma_bounds, method='bounded', options={'xatol':1e-3})
    sigma0 = float(res.x)

    # refine by searching in a narrower window around sigma0
    low = max(sigma_bounds[0], sigma0 - 0.8)
    high = min(sigma_bounds[1], sigma0 + 0.8)

    def obj_ref(s):
        return -compute_h(s, xi, alpha, N=N_refine)

    res2 = minimize_scalar(obj_ref, bounds=(low, high), method='bounded', options={'xatol':1e-4})
    sigma_opt = float(res2.x)
    h_max = compute_h(sigma_opt, xi, alpha, N=N_refine)
    return sigma_opt, h_max


def optimal_T1(epsilon, Gamma, P_in):
    """T1_opt = epsilon/2 + sqrt((epsilon/2)^2 + Gamma * P_in)"""
    return 0.5 * epsilon + np.sqrt((0.5 * epsilon)**2 + Gamma * P_in)


def intracavity_power(T1, epsilon, Gamma_eff, P_in):
    """
    Solve Pc = T1*P_in / (T1 + epsilon + Gamma_eff*Pc)^2 robustly.
    Return np.nan on failure.
    """
    def f(Pc):
        denom = (T1 + epsilon + Gamma_eff * Pc)
        return Pc - (T1 * P_in) / (denom * denom)

    # bracket: [0, P_up]
    P_up = max(1e3 * P_in, 1.0)
    try:
        # ensure signs differ
        fa = f(0.0)
        fb = f(P_up)
        if np.sign(fa) == np.sign(fb):
            # try expanding the bracket a bit
            P_up *= 1e3
            fb = f(P_up)
            if np.sign(fa) == np.sign(fb):
                return np.nan
        Pc = brentq(f, 0.0, P_up, maxiter=200)
        return Pc
    except Exception:
        return np.nan


def impedance_match_intravavity_power(P_in, T1_opt):
    """
    Compute intracavity power at impedance matching.
    :param P_in:
    :param T1_opt:
    :return:
    """
    return P_in / T1_opt