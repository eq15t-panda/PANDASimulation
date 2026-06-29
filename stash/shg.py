import matplotlib.pyplot as plt
import numpy as np

import utils.plot_parameters as pp


def out_shg_sorensen(input_coupler, loss, input_power, nonlinear_efficiency):
    """
    Second-harmonic output power from Sorensen 1998
    :param input_coupler:
    :param loss:
    :param input_power:
    :param nonlinear_efficiency:
    :return:
    """
    a = input_coupler + loss
    rho = 4 * input_coupler * input_power * nonlinear_efficiency

    prefactor = a**2 / (9 * nonlinear_efficiency)

    b = 1 + (27 / 2) * (rho / a**3) * (1 + np.sqrt(1 + (4 / 27) * (a**3 / rho)))

    res = prefactor * ((b ** (1 / 6)) - (b ** (-1 / 6)))**4
    return res


# Parameters
loss = 0.02
nonlinear_efficiency = 0.02

plt.figure(figsize=(12, 5))
# Subplot (a): Output SHG vs input coupler for various input powers
plt.subplot(1, 2, 1)
input_powers = [0.1, 0.2, 0.4, 0.6, 0.8]  # in Watts
input_couplers = np.linspace(0.01, 0.3, 300)
colors = plt.cm.viridis(np.linspace(0, 1, len(input_powers)))
optimal_T1_list = []

for idx, P1 in enumerate(input_powers):
    outputs = [out_shg_sorensen(T1, loss, P1, nonlinear_efficiency) for T1 in input_couplers]
    outputs = np.array(outputs)
    plt.plot(input_couplers * 100, outputs * 1e3, label=f'{int(P1 * 1e3)} mW', color=colors[idx])

    max_index = np.argmax(outputs)
    optimal_T1 = input_couplers[max_index]
    optimal_T1_list.append(optimal_T1)

    # Plot the max point as a dot
    plt.plot(optimal_T1 * 100, outputs[max_index] * 1e3, 'o', color=colors[idx], markersize=6)

# Highlight ±10% region around last optimal T1
# optimal_T1_percent = optimal_T1_list[-1] * 100
# lower_bound = optimal_T1_percent * 0.9
# upper_bound = optimal_T1_percent * 1.1
# plt.axvline(lower_bound, color='gray', linestyle='--', linewidth=1)
# plt.axvline(upper_bound, color='gray', linestyle='--', linewidth=1)
# plt.axvspan(lower_bound, upper_bound, color='gray', alpha=0.2,
#             label=f'±10% around {optimal_T1_percent:.2f}%')

plt.xlabel('Input Coupler Transmission (%)')
plt.ylabel('Output Power (mW)')
# plt.title('(a) SHG Output vs Input Coupler')
plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
plt.grid(True)

# Subplot (b): Output SHG vs input power for each optimal input coupler
plt.subplot(1, 2, 2)
input_powers_cont = np.linspace(start=0.01, stop=1.0, num=300)

for idx, (P_fixed, T1_opt) in enumerate(zip(input_powers, optimal_T1_list)):
    outputs_b = [out_shg_sorensen(T1_opt, loss, p, nonlinear_efficiency) for p in input_powers_cont]
    plt.plot(input_powers_cont * 1000, [P2 * 1e3 for P2 in outputs_b], label=f'Opt @ {int(P_fixed * 1e3)} mW = {T1_opt*100:.2f}%', color=colors[idx])

plt.xlabel('Input Power (mW)')
plt.ylabel('Output Power (mW)')
# plt.title('(b) SHG Output vs Input Power')
plt.grid(True)
plt.legend(fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.show()
