import seaborn as sns


def set_style(style=None):
    """Set basic plotting style and fonts."""
    if style is None:
        style = (
            "ticks",
            {
                "font.family": "sans-serif",
                "font.serif": ["Arial"],
                "font.size": 16,
                "axes.grid": True,
            },
        )
    else:
        style = style
    sns.set_style(*style)


def rgb2hex(r, g, b):
    """RGB to hexadecimal."""
    return "#%02x%02x%02x" % (r, g, b)

