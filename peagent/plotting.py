"""Publication plotting defaults (Illustrator-compatible fonts, black frame)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

FONT_ORDER = ["Arial", "Arial Unicode MS", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


def set_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": FONT_ORDER,
        "axes.titlesize": 18, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.grid": False, "axes.linewidth": 1.2, "axes.edgecolor": "black",
    })


def frame(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.2)
    return ax


def save(fig, prefix):
    """Export both vector PDF and 600-dpi PNG with a shared prefix."""
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
