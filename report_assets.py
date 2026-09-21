from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def make_schematic(output: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    t = np.linspace(0, 0.64, 321)
    # Deterministic illustrative paths chosen to show a first and a corrective crossing.
    a = 0.06 + 1.88 * t + 0.055 * np.sin(22 * t) + 0.025 * np.sin(51 * t)
    b = 0.02 + 1.33 * t + 0.06 * np.sin(18 * t + 1.4) + 0.02 * np.sin(43 * t)
    threshold = 0.82
    first = int(np.argmax(a >= threshold))
    # Give the losing accumulator a late push so it crosses within the response window.
    b = b + np.clip(t - t[first], 0, None) * 1.15
    second_candidates = np.flatnonzero((b >= threshold) & (np.arange(len(t)) > first))
    second = int(second_candidates[0])

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=220)
    ax.axhline(threshold, color="#20242a", lw=1.6, ls="--")
    ax.text(0.012, threshold + 0.025, "Common response threshold  b", color="#20242a")
    ax.plot(t, a, color="#1d5f91", lw=2.6, label=r"Accumulator A (drift $v_A$)")
    ax.plot(t, b, color="#b55232", lw=2.6, label=r"Accumulator B (drift $v_B$)")
    ax.axvline(t[first], color="#1d5f91", lw=1.2, alpha=0.8)
    ax.axvline(t[second], color="#b55232", lw=1.2, alpha=0.8)
    ax.axvspan(t[first], t[first] + 0.25, color="#d4a72c", alpha=0.13)
    ax.scatter([t[first]], [a[first]], s=55, color="#1d5f91", zorder=5)
    ax.scatter([t[second]], [b[second]], s=55, color="#b55232", zorder=5)
    ax.annotate("First response", (t[first], a[first]), xytext=(t[first] - 0.13, 1.08),
                arrowprops=dict(arrowstyle="->", color="#1d5f91"), color="#1d5f91", fontweight="bold")
    ax.annotate("Opposite response\nwithin 250 ms", (t[second], b[second]), xytext=(t[second] + 0.035, 1.02),
                arrowprops=dict(arrowstyle="->", color="#b55232"), color="#b55232", fontweight="bold")
    ax.annotate(r"Mutual inhibition  $\beta$", xy=(0.30, 0.58), xytext=(0.30, 0.31), ha="center",
                arrowprops=dict(arrowstyle="<->", color="#665c54"), color="#665c54", fontweight="bold")
    ax.text((t[first] + min(t[first] + 0.25, t[-1])) / 2, 0.075, "double-response window", ha="center",
            color="#775b00", fontweight="bold")
    ax.set_xlim(0, 0.64)
    ax.set_ylim(0, 1.20)
    ax.set_xlabel("Decision time (s); observed RT adds non-decision time $t_0$")
    ax.set_ylabel("Evidence")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_title("Illustration of a double response in the lateral-inhibition racing diffusion model",
                 loc="left", pad=14, fontweight="bold", color="#17253d")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


make_schematic(OUT / "model_schematic.png")
print(f"Saved {OUT / 'model_schematic.png'}")