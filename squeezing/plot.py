import matplotlib.pyplot as plt
import numpy as np

from squeezing.noise_spectrum import noise_spectrum_x, noise_spectrum_p
from utils.settings import settings
import utils.plot_style as pm

# -- Arrays definitions -- #
lambda_array = np.linspace(start=0, stop=1000, num=settings.number_points) * 1e-9  # wavelength
omega_array = np.linspace(start=0, stop=3*settings.omega_c, num=settings.number_points)

escape_efficiencies = np.linspace(start=0.8, stop=0.96, num=10)  # escape efficiency range
epsilon_array = np.linspace(start=0, stop=1, num=settings.number_points, endpoint=False)  # threshold or pump power

colors = plt.cm.viridis(np.linspace(0, 1, len(escape_efficiencies)))  # Using a colormap for colors


# -- Squeezing and anti-squeezing versus pump power -- #
def squeezing_vs_pump(omega=0.0):
    fig_pump_power, ax = plt.subplots(figsize=(16, 9))
    for idx, escape_efficiency in enumerate(escape_efficiencies):
        sx = noise_spectrum_x(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon_array)
        sp = noise_spectrum_p(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon_array)
        color = colors[idx]
        ax.plot(epsilon_array**2, 10*np.log10(sx), color=colors[idx])
        ax.plot(epsilon_array**2, 10*np.log10(sp), color=colors[idx], linestyle='--')

    # Create the legend (for line style)
    legend_quadrature = ax.legend(['$s_x$', '$s_p$'],
                                  loc='upper left',
                                  bbox_to_anchor=(1.02, 1.02),
                                  title='Quadratures',
                                  title_fontsize=pm.MEDIUM_SIZE-5)

    for line in legend_quadrature.get_lines():
        line.set_color('black')  # Set the legend lines to black

    # Enforce parameters for legend titles
    legend_quadrature._legend_box.align = "left"  # adjust alignment
    # legend_quadrature._legend_box.sep = 15  # change title padding

    ax.add_artist(legend_quadrature)

    # Create custom legend handles for escape efficiencies
    legend_eta = ax.legend([(color, '--') for color in colors],
                           [f'${eta:.3f}$' for eta in escape_efficiencies],
                           handler_map={tuple: pm.AnyObjectHandler()},
                           loc='lower left',
                           bbox_to_anchor=(1.02, 0),
                           title='Escape efficiency',
                           title_fontsize=pm.MEDIUM_SIZE-5,
                           fontsize=18)

    legend_eta._legend_box.align = "left"  # adjust alignment
    # legend_eta._legend_box.sep = 15  # change title padding
    ax.add_artist(legend_eta)

    # Set labels
    ax.set_xlabel("$P/P_{thr} = \epsilon^2$")
    ax.set_ylabel('$S$ (dB)')
    # ax.set_title(f'Squeezing and anti-squeezing versus pump power at zero frequency.', pad=20)

    # Adjust the layout to accommodate both legends
    plt.subplots_adjust(right=0.8)
    plt.grid(True)
    plt.show()


# -- Squeezing and anti-squeezing versus wavelength -- #
def squeezing_vs_wavelength(escape_efficiency=0.9, epsilon=0.1):
    """

    :param escape_efficiency:
    :param epsilon:
    :return:
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    sx = noise_spectrum_x(omega=omega_array, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon)
    sp = noise_spectrum_p(omega=omega_array, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon)
    plt.plot(omega_array/settings.omega_c, 10*np.log10(sx), color='k')
    plt.plot(omega_array/settings.omega_c, 10*np.log10(sp), color='k', linestyle='--')

    # Add some point of interest

    # Create the legend (for line style)
    legend_quadrature = plt.legend(['$s_x$', '$s_p$'], loc='upper left', bbox_to_anchor=(1.0, 1), title='Quadratures')

    plt.xlabel("$\omega/\omega_c$")
    plt.ylabel('$S$ (dB)')

    # Add some legend
    text = f'$\eta =$ {escape_efficiency}\n$\epsilon =$ {epsilon}'
    # plt.annotate(text,
    #           xy=(1.02, 0.6),
    #           xycoords='axes fraction',
    #           size=18,
    #           bbox=dict(boxstyle="round,pad=1", edgecolor='black', fc='none'))

    # Create the legend (for line style)
    parameters_legend = ax.legend([f'$\eta=${escape_efficiency}', f'$\epsilon=${epsilon}'],
                                  loc='upper left',
                                  bbox_to_anchor=(1.0025, 0.7),
                                  handlelength=0,
                                  handletextpad=0,
                                  fontsize=18)
    for item in parameters_legend.legendHandles:
        item.set_visible(False)

    ax.add_artist(legend_quadrature)
    ax.add_artist(parameters_legend)

    # Adjust the layout to accommodate both legends
    plt.subplots_adjust(right=0.85)
    plt.grid(True)
    plt.show()
