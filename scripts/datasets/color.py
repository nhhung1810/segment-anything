from matplotlib.colors import to_rgba, get_named_colors_mapping

# from matplotlib.colormaps import get_cmap
import matplotlib


key_names = list(get_named_colors_mapping().keys())
num_key = len(key_names)


def get_color(idx, alpha=None):
    """Make RGBA color tuple from idx.
    This method is deterministic

    Args:
        idx (int): A number

    Returns:
        Tuple: (R, G, B, A)
    """
    (r, g, b, a) = matplotlib.colormaps[SEQUENTIAL[idx % len(QUALITATIVE)]](0.9)
    if not alpha:
        return (r, g, b, a)
    return (r, g, b, alpha)


# fmt: off
# QUALITATIVE colormap
QUALITATIVE = ["Pastel1","Pastel2","Paired","Accent","Dark2","Set1","Set2","Set3","tab10","tab20","tab20b","tab20c"]

SEQUENTIAL = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]
# fmt: on

if __name__ == "__main__":
    # to_rgba(QUALITATIVE)
    # get_cmap(QUALITATIVE[0])
    print(matplotlib.colormaps[QUALITATIVE[0]](0.5))
    pass
