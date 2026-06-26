import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

from numpy import vstack, linspace

# ------------------------------------------------
# Document geometry — derived from your preamble
# A4=210mm, margins=2×25mm, binding=10mm
# textwidth = 210 - 50 - 10 = 150mm
# ------------------------------------------------
_mm_to_in = 1.0 / 25.4
_textwidth = 150.0 * _mm_to_in  # 5.906 inches — full \textwidth
_golden = (1.0 + 5.0 ** 0.5) / 2.0  # 1.618 — default aspect ratio


def fig_size(width_frac=1.0, aspect=None):
    """
    Return (width, height) in inches so that matplotlib's font sizes
    match LaTeX's after $\includegraphics[width=<width_frac>\textwidth]{...}$.

    width_frac : fraction of \textwidth  (1.0 = full, 0.5 = side-by-side)
    aspect     : height/width ratio. Defaults to 1/golden.
    """
    w = _textwidth * width_frac
    h = w / _golden if aspect is None else w * aspect
    return w, h


custom = {
    "blue390": "#007dff",
    "red780": "#ff0000",
}


def set_plot_style(width_frac=1.0, aspect=None):
    # Add custom colors to CSS4_COLORS for easy use by name
    mcolors._colors_full_map.update(custom)

    # Set plot style parameters to match LaTeX document style
    mpl.rcParams.update({

        # ------------------------------------------------
        # Typography
        # ------------------------------------------------
        "text.usetex": True,
        "text.latex.preamble":
            r"\usepackage{libertine}"
            r"\usepackage[libertine]{newtxmath}",

        "font.family": "serif",

        # Slightly larger than document body for readability
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,

        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "legend.fontsize": 12,

        # ------------------------------------------------
        # Figure geometry
        # ------------------------------------------------
        "figure.figsize": fig_size(width_frac, aspect),

        # Higher preview DPI helps notebook rendering
        "figure.dpi": 180,

        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        # ------------------------------------------------
        # Axes
        # ------------------------------------------------
        "axes.linewidth": 0.9,
        "axes.labelpad": 6,

        # Cleaner scientific style
        "axes.spines.top": True,
        "axes.spines.right": True,

        # ------------------------------------------------
        # Ticks
        # ------------------------------------------------
        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "xtick.major.size": 5,
        "ytick.major.size": 5,

        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,

        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # ------------------------------------------------
        # Grid
        # ------------------------------------------------
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,

        # ------------------------------------------------
        # Lines
        # ------------------------------------------------
        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        # ------------------------------------------------
        # Color cycle — link blue + cite purple anchor
        # ------------------------------------------------
        "axes.prop_cycle": mpl.cycler(color=[
            "#0072CF",  # link blue
            "#e05c2a",  # burnt orange
            "#2ca02c",  # green
            "#7818c8",  # cite purple
            "#d62728",  # red
            "#17becf",  # teal
            "#8c564b",  # brown
            "#7f7f7f",  # gray
        ]),
        # ------------------------------------------------
        # Legend
        # ------------------------------------------------
        "legend.frameon": True,

        # ------------------------------------------------
        # Better scientific default colormap
        # ------------------------------------------------
        "image.cmap": "viridis",
    })


class AnyObjectHandler(HandlerBase):

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height], linestyle=orig_handle[1], color=orig_handle[0])
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height], color=orig_handle[0])
        return [l1, l2]


def symmetrical_colormap(cmap_settings, new_name=None):
    """
    This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold.
    :param cmap_settings:
    :param new_name:
    :return:
    """
    # get the colormap
    current_cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # this defined the roughness of the colormap, 128 is fine
    n = 256

    # get the list of color from colormap
    colors_r = current_cmap(linspace(start=0, stop=1, num=n))  # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = vstack((colors_r, colors_l))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap
