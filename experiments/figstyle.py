"""Shared matplotlib style for all thesis figures (print-safe: colour AND line style/marker differ)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "grey": "#6b6b6b", "ink": "#0b0b0b", "soft": "#52514e",
     "yellow": "#eda100", "violet": "#6250d6", "red": "#e34948", "magenta": "#e87ba4"}
SERIES = [C["blue"], C["orange"], C["aqua"], C["grey"]]
LS = ["-", "--", "-.", ":"]
MK = ["o", "s", "^", "D"]


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.edgecolor": C["soft"], "axes.labelcolor": C["ink"], "text.color": C["ink"], "xtick.color": C["soft"], "ytick.color": C["soft"],
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
        "lines.linewidth": 1.8,
    })
