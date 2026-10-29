import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from numpy import vstack, linspace

from utils.plot_helper import cmap_map

# ------------------------------------------------
# Document geometry — must match your .tex preamble exactly
# \documentclass[a4paper,11pt]{book}
# \usepackage[margin=28mm,includeheadfoot]{geometry}
# textwidth = 210 - 2×28 = 154mm  (includeheadfoot doesn't affect width)
# ------------------------------------------------
_mm_to_in = 1.0 / 25.4
_textwidth = 149.0 * _mm_to_in  # 5.906 inches — full \textwidth
PANEL_W = (149.0 - 5.0) / 2 * _mm_to_in  # 72mm — half-width with 5mm gutter
_golden = (1.0 + 5.0 ** 0.5) / 2.0  # 1.618 — default aspect ratio

# ------------------------------------------------
# LaTeX font sizes at 11pt book class
# Use these explicitly — never hardcode a number
# ------------------------------------------------
NORMALSIZE = 10.95  # \normalsize
SMALL = 10.0    # \small
FOOTNOTESIZE = 9.0    # \footnotesize
SCRIPTSIZE = 8.0    # \scriptsize

custom = {
    "blue390": "#007dff",
    "red780": "#ff0000",
}

# ------------------------------------------------
# Custom color maps
# ------------------------------------------------
colors = ["orange", "yellow", "white", "lightblue", "darkblue"]

# Diverging: purple -> white -> green
diverging_cmap = LinearSegmentedColormap.from_list(
    "custom_diverging",
    [
        (118/255, 42/255, 131/255),   # purple
        (247/255, 247/255, 247/255),  # white
        (27/255,  120/255, 55/255),   # green
    ]
)

# Sequential: cream -> orange -> dark brown
sequential_cmap = LinearSegmentedColormap.from_list(
    "custom_sequential",
    [
        (255/255, 225/255, 229/255),  # cream
        (251/255, 154/255, 41/255),   # orange
        (102/255, 37/255,  6/255),    # dark brown
    ]
)


# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def fig_size(width_frac=1.0, aspect=None):
    """
    Return (width, height) in inches so that matplotlib's font sizes
    match LaTeX's after $includegraphics[width=<width_frac>\textwidth]{...}$.

    width_frac : fraction of \textwidth  (1.0 = full, 0.5 = side-by-side)
    aspect     : height/width ratio. Defaults to 1/golden.
    """
    w = _textwidth * width_frac
    h = w / _golden if aspect is None else w * aspect
    return w, h


def set_plot_style(width_frac=1.0, aspect=None):
    # Add custom colors to CSS4_COLORS for easy use by name
    mcolors._colors_full_map.update(custom)

    # Set plot style parameters to match LaTeX document style
    mpl.rcParams.update({

        # ------------------------------------------------
        # Typography — match document exactly
        # ------------------------------------------------
        "text.usetex": True,
        "text.latex.preamble":
            r"\usepackage{libertine}"
            r"\usepackage[libertine]{newtxmath}",
        "font.family": "serif",

        # Hierarchy
        "font.size": NORMALSIZE,  # 10.95 — base
        "axes.titlesize": NORMALSIZE,  # same level as body text
        "axes.labelsize": SMALL,  # 10pt — slightly subordinate
        "xtick.labelsize": FOOTNOTESIZE,  # 9pt  — clearly subordinate
        "ytick.labelsize": FOOTNOTESIZE,  # 9pt
        "legend.fontsize": FOOTNOTESIZE,  # 9pt

        # ------------------------------------------------
        # Figure geometry
        # ------------------------------------------------
        "figure.figsize": fig_size(width_frac, aspect),
        # Higher preview DPI helps notebook rendering
        "figure.dpi": 180,

        "savefig.dpi": 400,
        # "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,

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
        "xtick.direction":      "in",
        "ytick.direction":      "in",

        "xtick.top":            True,
        "ytick.right":          True,

        "xtick.major.size":     4.0,
        "ytick.major.size":     4.0,
        "xtick.minor.size":     2.0,
        "ytick.minor.size":     2.0,

        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.width":    0.5,
        "ytick.minor.width":    0.5,

        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,

        # ------------------------------------------------
        # Grid
        # ------------------------------------------------
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,

        # ------------------------------------------------
        # Lines
        # ------------------------------------------------
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,

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
