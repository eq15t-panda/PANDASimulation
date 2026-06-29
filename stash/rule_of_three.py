import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from nlo.sellmeier import n_z
from utils.tools_db import load_all_data
from utils.settings import settings
import utils.plot_style


def safe_divide(array, value):
    try:
        if value is None or not np.isscalar(value):
            raise ValueError
        return array / float(value)
    except (KeyError, TypeError, ValueError):
        return np.full_like(array, np.nan, dtype=np.float64)


def safe_multiply(*args):
    def safe_to_float(arg):
        try:
            return np.float64(arg)
        except (ValueError, TypeError):
            return np.nan

    args = [safe_to_float(arg) for arg in args]
    result = args[0]
    for arg in args[1:]:
        result = np.where(np.isnan(result) | np.isnan(arg), np.nan, result * arg)
    return result


def plot_squeezing_1d():
    db_data = load_all_data()
    shg_entries, opo_entries = {}, {}

    for author, subsystems in db_data.items():
        for system_name, entries in subsystems.items():
            if not entries:
                continue
            key = f"{author}:{system_name.lower()}"
            if "shg" in system_name.lower():
                shg_entries[key] = entries
            elif "opo" in system_name.lower():
                opo_entries[key] = entries

    authors = [key.split(":")[0] for key in opo_entries.keys()]
    author_colors = {author: plt.cm.tab10(i % 10) for i, author in enumerate(authors)}

    # Set our experiment parameters
    wavelength = 0.780  # µm
    input_power = np.linspace(start=20, stop=150, num=100)  # mW
    length_crystal = np.array([10, 20, 25, 30])  # mm
    roc = np.array([50, 75, 100, 150])
    output_coupler = np.array([0.01, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15])
    round_trip = np.linspace(start=450, stop=600, num=100)
    threshold_power = np.linspace(start=0, stop=400, num=100)

    index_780 = n_z(wavelength)

    fig = plt.figure()
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.2], hspace=0.4)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    caption_ax = fig.add_subplot(gs[2, :])
    caption_ax.axis("off")

    handles, labels = [], []

    for author in authors:
        key = f"{author}:opo"
        if key not in opo_entries:
            continue
        opo = opo_entries[key][0]

        wavelength_ref = opo["input_wavelength_nm"] * 1e-3
        index_ratio = index_780 / n_z(wavelength_ref * 2)
        S = 10 ** (opo["squeezing_dB"] / 10)

        # Reference values
        power_ratio = safe_divide(input_power, opo.get("input_power_mW"))
        crystal_ratio = safe_divide(length_crystal, opo.get("crystal_length_mm"))
        coupler_ratio = safe_divide(output_coupler, opo.get("T_output_coupler"))
        roc_ratio = safe_divide(roc, opo.get("roc1_mm"))
        round_trip_ratio = safe_divide(round_trip, opo.get("cavity_length_mm"))
        threshold_power_ratio = safe_divide(threshold_power, opo.get("threshold_power_mW"))

        sq_power = safe_multiply(S, index_ratio, power_ratio)
        sq_crystal = safe_multiply(S, index_ratio, crystal_ratio)
        sq_coupler = safe_multiply(S, index_ratio, coupler_ratio)
        sq_roc = safe_multiply(S, index_ratio, roc_ratio)
        sq_round_trip = safe_multiply(S, index_ratio, round_trip_ratio)
        sq_threshold_power = safe_multiply(S, index_ratio, threshold_power_ratio)

        color = author_colors[author]
        line1, = axes[0].plot(input_power, sq_power, label=author, color=color)
        line2, = axes[1].plot(length_crystal, sq_crystal, '-s', label=author, color=color)
        line3, = axes[2].plot(output_coupler, sq_coupler, label=author, color=color)
        line4, = axes[3].plot(roc, sq_roc, '-s', label=author, color=color)
        line5, = axes[4].plot(round_trip, sq_round_trip, label=author, color=color)
        line6, = axes[5].plot(threshold_power, sq_threshold_power, label=author, color=color)

        def get_value_or_default(opo, key, default='\u2297'):
            value = opo.get(key)
            return value if value is not None else default

        legend_label = (
            fr"$\mathbf{{{author}}}$" + "\n"
            fr"$\lambda={wavelength_ref:.3f}~\mu\text{{m}}$" + "\n"
            fr"$P={get_value_or_default(opo, 'input_power_mW')}~\text{{mW}}$, "
            fr"$l_c={get_value_or_default(opo, 'crystal_length_mm')}~\text{{mm}}$" + "\n"
            fr"$T={get_value_or_default(opo, 'T_output_coupler')}$, "
            fr"$\text{{ROC}}={get_value_or_default(opo, 'roc1_mm')}~\text{{mm}}$" + "\n"
            fr"$\text{{L}}={get_value_or_default(opo, 'round_trip_mm')}~\text{{mm}}$, "
            fr"$P_{{thr}}={get_value_or_default(opo, 'threshold_power_mW')}~\text{{mW}}$"
        )
        if author not in labels:
            handles.append(line1)
            labels.append(legend_label)

    titles = [
        "1) Squeezing vs. Input Power (mW)",
        "2) Squeezing vs. Crystal Length (mm)",
        "3) Squeezing vs. Output Coupler",
        "4) Squeezing vs. Mirror ROC (mm)",
        "5) Squeezing vs. Round trip length (mm)",
        "6) Squeezing vs. Pump Threshold (mW)"
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=16)

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=len(authors), fontsize=12)
    # plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()


def plot_squeezing_contour(ax, param1, param2, param1_vals, param2_vals, opo, index_ratio, title):
    X, Y = np.meshgrid(param1_vals, param2_vals)
    S = 10 ** (opo["squeezing_dB"] / 10)

    def extract_ref(pname):
        return opo.get({
            "input_power": "input_power_mW",
            "crystal_length": "crystal_length_mm",
            "output_coupler": "T_output_coupler",
            "roc": "roc1_mm",
            "cavity_length": "cavity_length_mm"
        }[pname], np.nan)

    X_ratio = safe_divide(X, extract_ref(param1))
    Y_ratio = safe_divide(Y, extract_ref(param2))

    squeezing = safe_multiply(S, index_ratio, X_ratio, Y_ratio)

    cp = ax.pcolormesh(X, Y, squeezing, shading='auto', cmap='plasma')
    ax.set_xlabel(param1.replace("_", " ").title())
    ax.set_ylabel(param2.replace("_", " ").title())
    ax.set_title(title)

    return cp


# Run both
if __name__ == "__main__":
    plot_squeezing_1d()
    # Example contour call for Burks with different parameter pairs
    # db_data = load_all_data()
    # opo_entries = {f"{a}:{s.lower()}": e for a, subs in db_data.items() for s, e in subs.items() if "opo" in s.lower()}
    #
    # burks_key = "Burks:opo"
    # if burks_key in opo_entries:
    #     opo = opo_entries[burks_key][0]
    #
    #     index_780 = n_z(0.780)
    #     index_ratio = index_780 / n_z(opo["input_wavelength_nm"] * 2 * 1e-3)
    #
    #     # Define parameter pairs and their values
    #     parameter_pairs = [
    #         ("input_power", "output_coupler", np.linspace(20, 150, 100), np.linspace(0.01, 0.15, 100)),
    #         ("crystal_length", "roc", np.linspace(10, 30, 100), np.linspace(50, 150, 100)),
    #         ("input_power", "roc", np.linspace(20, 150, 100), np.linspace(50, 150, 100))
    #     ]
    #
    #     # Create a figure with subplots
    #     fig, axes = plt.subplots(1, len(parameter_pairs), figsize=(20, 5))
    #
    #     for ax, (param1, param2, param1_vals, param2_vals) in zip(axes, parameter_pairs):
    #         cp = plot_squeezing_contour(
    #             ax=ax,
    #             param1=param1,
    #             param2=param2,
    #             param1_vals=param1_vals,
    #             param2_vals=param2_vals,
    #             opo=opo,
    #             index_ratio=index_ratio,
    #             title=f"Squeezing vs. {param1.replace('_', ' ').title()} and {param2.replace('_', ' ').title()}"
    #         )
    #
    #     # Add a colorbar to the figure
    #     fig.colorbar(cp, ax=axes[len(axes) - 1], orientation='vertical', fraction=0.1, pad=0.04,
    #                  label="Estimated Squeezing")
    #
    #     # Adjust the layout to make room for the colorbar and prevent overlapping
    #     plt.subplots_adjust(wspace=0.3, hspace=0.1)  # Adjust the spacing between subplots
    #     plt.tight_layout()  # Adjust the right margin to make room for the colorbar
    #     plt.show()
    #
    # else:
    #     print("Burks data not found.")
