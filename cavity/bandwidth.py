from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

import sqlite3

from utils.settings import settings
import utils.plot_style as mplp
from utils.tools_db import load_all_data

import cavity.cavity_formulas as cf


def bandwidth(bandwidth_list=np.array([6, 10, 20])*1e6):
    # Setting parameters
    cavity_lengths = np.linspace(start=settings.min_L, stop=settings.max_L, num=settings.number_points)
    transmission_coefficients = np.linspace(start=settings.min_T, stop=settings.max_T, num=settings.number_points)

    L, T = np.meshgrid(cavity_lengths, transmission_coefficients)
    bandwidth_meshgrid = cf.Bandwidth_bowtie(T=T, L=L, Loss=settings.cavity_loss) * 1e-6

    # Define the colours and colormaps for the bandwidths
    bandwidth_colours = ['lightsteelblue', 'cornflowerblue', 'royalblue']
    bandwidth_cmaps = [mplp.symmetrical_colormap((x, None)) for x in ['Blues', 'Greens', 'Reds']]

    # Bandwidth surface plot
    clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.2)

    fig_bandwidth = plt.figure(figsize=(16, 9))
    # Creating three axes: add_axes([xmin,ymin,dx,dy])
    ax1 = fig_bandwidth.add_axes((0.75, 0.1, 0.1, 0.28))
    ax2 = fig_bandwidth.add_axes((0.75, 0.395, 0.1, 0.28))
    ax3 = fig_bandwidth.add_axes((0.75, 0.69, 0.1, 0.28))
    ax_bandwidth = fig_bandwidth.add_axes((0.1, 0.1, 0.7, 0.87))
    axes = [ax1, ax2, ax3]

    for central_freq, colour, selected_cmap, axis in zip(bandwidth_list, bandwidth_colours, bandwidth_cmaps, axes):

        # Find couple (L, T) such that Delta within range
        boundary_down = (central_freq - settings.range_freq * central_freq) * 1e-6
        boundary_up = (central_freq + settings.range_freq * central_freq) * 1e-6
        indices = np.where((bandwidth_meshgrid < boundary_up) & (
                    bandwidth_meshgrid > boundary_down))  # here np.where gives a tuple, the first element of which gives the row index, while the

        # Mask the region outside the desired range
        mask = np.ones_like(bandwidth_meshgrid, dtype=bool)
        mask[indices] = False
        masked_bandwidth = np.ma.masked_array(bandwidth_meshgrid, mask)

        # Plot the masked region with a different colormap
        contour_zone = ax_bandwidth.contourf(L, T, masked_bandwidth, cmap=selected_cmap, alpha=settings.alpha)
        axis.axis('off')
        cbar_zone = plt.colorbar(contour_zone, ax=axis)
        cbar_zone_ticks = cbar_zone.get_ticks()
        cbar_zone.set_ticks(cbar_zone_ticks[1:-1])

    # Add some points to give couples from different articles
    # Fetch data from the database
    data = load_all_data()

    # Plot SHG and SPDC data
    index = 1
    shg_legend_elements = []
    spdc_legend_elements = []

    shg_marker = "o"
    spdc_marker = "*"

    for author, entries in data.items():
        # Plot SHG entries
        for shg_entry in entries["shg"]:
            x, y = shg_entry["cavity_length_mm"] / 1000, shg_entry["T_input_coupler"]
            ax_bandwidth.scatter(x, y, s=90, marker=shg_marker, color="black")
            ax_bandwidth.text(x + 5e-3, y + 0.002, f"{index}", color="black", fontsize=9, ha='left', va='bottom')
        shg_legend_elements.append(Line2D([0], [0], marker=shg_marker, color="black", label=f"{index}. {author}", linestyle=""))

        # Plot SPDC entries
        for opo_entry in entries["opo"]:
            x, y = opo_entry["cavity_length_mm"] / 1000, opo_entry["T_output_coupler"]
            ax_bandwidth.scatter(x, y, s=90, marker=spdc_marker, color="black")
            ax_bandwidth.text(x + 5e-3, y + 0.002, f"{index}", color="black", fontsize=9, ha='left', va='bottom')
        spdc_legend_elements.append(Line2D([0], [0], marker=spdc_marker, color="black", label=f"{index}. {author}", linestyle=""))

        index += 1

    legend_elements = shg_legend_elements + spdc_legend_elements

    # Create a text box for authors
    authors_text = "\n".join([f"{index}. {author}" for index, author in enumerate(data.keys(), start=1)])
    authors_box = AnchoredText(
        authors_text,
        loc="upper left",  # Adjust position as needed
        prop={'size': 15},
        frameon=True,  # Add a box around the text
        pad=0.5,
        borderpad=1
    )
    ax_bandwidth.add_artist(authors_box)  # Add the text box to the plot

    # Create a legend for the markers (SHG and SPDC)
    marker_legend_elements = [
        Line2D([0], [0], marker=shg_marker, color="black", label="SHG", linestyle="none"),
        Line2D([0], [0], marker=spdc_marker, color="black", label="SPDC", linestyle="none")
    ]

    ax_bandwidth.legend(
        handles=marker_legend_elements,
        loc="upper right",  # Adjust position as needed
        prop={'size': 15},
        frameon=True,  # Add a box around the legend
        title="Markers"
    )
    # Set axis labels
    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')

    # Add text vertically on the right side of the axes
    text_y_pos = 0.5  # Vertically centered position
    text_x_pos = 0.92  # Right side position
    text_vertical = "Bandwidth Δ (MHz)"
    plt.text(text_x_pos, text_y_pos, text_vertical, horizontalalignment='center', verticalalignment='center',
             rotation=270, transform=fig_bandwidth.transFigure, fontsize=mplp.MEDIUM_SIZE)

    plt.show()


if __name__ == "__main__":
    bandwidth()

    # T = np.array([0.05, 0.07, 0.10, 0.12]) # Transmission coefficients
    # T_name = ["5", "7", "10", "12"]  # Names for the transmission coefficients
    # L = np.linspace(start=0.4, stop=1.5, num=1000) # Cavity lengths in meters
    #
    # # Define the target bandwidth range
    # target_bandwidth = 6.0  # 6 MHz
    # tolerance = 0.005
    # bandwidth_min = target_bandwidth * (1 - tolerance)
    # bandwidth_max = target_bandwidth * (1 + tolerance)
    #
    # # Compute the bandwidth for each combination of T and L
    # results = {}
    # for t, t_name in zip(T, T_name):
    #     bandwidths = cf.Bandwidth_bowtie(T=t, L=L, Loss=0.) * 1e-6
    #     # print(bandwidths)
    #     # print("----------------------")
    #     valid_indices = np.where((bandwidths >= bandwidth_min) & (bandwidths <= bandwidth_max))[0]
    #     valid_L = L[valid_indices]
    #     results[t_name] = valid_L
    #
    # # Print the results
    # for t_name, valid_L in results.items():
    #
    #     print(f"T = {t_name}%:")
    #     if valid_L.size > 0:
    #         print(f"  Valid L values (m): {valid_L}")
    #     else:
    #         print("  No valid L values found.")
    #
    #     print("----------------------")
