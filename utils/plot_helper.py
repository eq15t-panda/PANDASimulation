from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib
import numpy as np


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def red_custom(z):
    n_colors = 20  # number of discrete levels

    custom_red = LinearSegmentedColormap.from_list(
        "optical_red",
        [
            (0.00, "#fff3f2"),
            (0.20, "#FFD3CC"),
            (0.40, "#FFA59B"),
            (0.60, "#FF968B"),
            (0.80, "#FF746C"),
            (1.00, "#FF4A3B"),
        ],
        N=256
    )

    custom_red_light = cmap_map(lambda x: x / 2 + 0.4, custom_red)

    vmin_rd = z.max()
    vmax_rd = z.min()

    cmap = ListedColormap(custom_red_light(np.linspace(0, 1, n_colors)))
    return cmap, vmin_rd, vmax_rd