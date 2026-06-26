from itertools import chain


def flip(items, ncol):
    """
    Flips the order of items in a list by grouping them into columns.
    *used to fill columns of legends first*

    Parameters:
        items (list): The list of items to be flipped.
        ncol (int): The number of columns to group the items into.

    Returns:
        itertools.chain: A chain object containing the flipped items.

    Source:
        https://stackoverflow.com/questions/10101141/
        matplotlib-legend-add-items-across-columns-instead-of-down
    """
    return chain(*[items[i::ncol] for i in range(ncol)])
