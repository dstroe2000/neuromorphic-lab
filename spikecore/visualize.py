"""Visualization utilities for SpikeCore compilation and simulation.

Provides:
  - Spike raster plots (neurons × timesteps)
  - Compilation graph visualization (partitioned Relay IR)
  - Weight distribution histograms
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def plot_spike_raster(
    spike_log: list[tuple[int, int]],
    num_cores: int | None = None,
    num_timesteps: int | None = None,
    title: str = "SpikeCore — Spike Raster",
    figsize: tuple[float, float] = (12, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot spike raster: neurons on y-axis, timesteps on x-axis.

    Args:
        spike_log: List of (timestep, core_id) tuples.
        num_cores: Total cores to display (auto-detected if None).
        num_timesteps: Total timesteps (auto-detected if None).
        title: Plot title.
        figsize: Figure size.
        ax: Optional axes to plot on.

    Returns:
        Figure if ax was None, else None.
    """
    if not spike_log:
        fig, ax_ = plt.subplots(figsize=figsize) if ax is None else (None, ax)
        ax_.text(0.5, 0.5, "No spikes recorded", ha="center", va="center",
                transform=ax_.transAxes, fontsize=14, color="gray")
        ax_.set_title(title)
        return fig

    timesteps_arr = np.array([s[0] for s in spike_log])
    core_ids = np.array([s[1] for s in spike_log])

    if num_timesteps is None:
        num_timesteps = int(timesteps_arr.max()) + 1
    if num_cores is None:
        num_cores = int(core_ids.max()) + 1

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(timesteps_arr, core_ids, s=4, c="black", marker="|", linewidths=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron Core ID")
    ax.set_title(title)
    ax.set_xlim(-0.5, num_timesteps - 0.5)
    ax.set_ylim(-0.5, num_cores - 0.5)
    ax.invert_yaxis()

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_compilation_graph(
    layer_names: list[str],
    layer_targets: list[str],
    layer_shapes: list[tuple[int, int]] | None = None,
    title: str = "Compilation Graph — Partitioned Layers",
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Visualize the partitioned compilation graph.

    Shows each layer as a box, colored by target (host vs SpikeCore).

    Args:
        layer_names: Name of each layer/operation.
        layer_targets: Target for each layer ("spikecore" or "host").
        layer_shapes: Optional (in, out) feature sizes per layer.
        title: Plot title.
        figsize: Figure size.
    """
    n = len(layer_names)
    fig, ax = plt.subplots(figsize=figsize)

    colors = {"spikecore": "#2196F3", "host": "#9E9E9E"}
    box_width = 0.8
    box_height = 0.6
    spacing = 1.2

    for i, (name, target) in enumerate(zip(layer_names, layer_targets)):
        x = i * spacing
        color = colors.get(target, "#CCCCCC")
        rect = mpatches.FancyBboxPatch(
            (x - box_width / 2, -box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x, 0.05, name, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white")
        ax.text(x, -0.15, f"[{target}]", ha="center", va="center", fontsize=7,
                color="white", alpha=0.9)

        if layer_shapes and i < len(layer_shapes):
            shape = layer_shapes[i]
            ax.text(x, -0.38, f"{shape[0]}→{shape[1]}", ha="center", fontsize=7,
                    color="#555555")

        # Arrow to next layer
        if i < n - 1:
            ax.annotate("", xy=((i + 1) * spacing - box_width / 2, 0),
                       xytext=(x + box_width / 2, 0),
                       arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2196F3", label="SpikeCore (offloaded)"),
        mpatches.Patch(color="#9E9E9E", label="Host CPU"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    ax.set_xlim(-1, n * spacing)
    ax.set_ylim(-0.8, 0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_weight_distribution(
    weights_int8: np.ndarray,
    layer_name: str = "Layer",
    figsize: tuple[float, float] = (8, 4),
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot histogram of quantized int8 weights.

    Args:
        weights_int8: Flattened int8 weight array.
        layer_name: Layer name for title.
        figsize: Figure size.
        ax: Optional axes.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.hist(weights_int8.flatten(), bins=np.arange(-128, 129) - 0.5,
            color="#1976D2", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Weight Value (int8)")
    ax.set_ylabel("Count")
    ax.set_title(f"Quantized Weight Distribution — {layer_name}")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

    stats_text = (
        f"mean={weights_int8.mean():.1f}\n"
        f"std={weights_int8.std():.1f}\n"
        f"range=[{weights_int8.min()}, {weights_int8.max()}]"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_comparison(
    pytorch_probs: np.ndarray,
    spikecore_scores: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "PyTorch vs SpikeCore — Output Comparison",
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Side-by-side comparison of PyTorch and SpikeCore outputs.

    Args:
        pytorch_probs: Softmax probabilities from PyTorch (10,).
        spikecore_scores: Raw output scores from SpikeCore (10,).
        class_names: Optional class labels.
        title: Plot title.
        figsize: Figure size.
    """
    n_classes = len(pytorch_probs)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(n_classes)
    width = 0.6

    # PyTorch
    pt_pred = np.argmax(pytorch_probs)
    colors_pt = ["#1976D2" if i != pt_pred else "#F44336" for i in range(n_classes)]
    ax1.bar(x, pytorch_probs, width, color=colors_pt, alpha=0.85)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"PyTorch (pred: {class_names[pt_pred]})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)

    # SpikeCore
    sc_norm = spikecore_scores.astype(float)
    sc_sum = sc_norm.sum()
    if sc_sum > 0:
        sc_norm = sc_norm / sc_sum
    sc_pred = np.argmax(spikecore_scores)
    colors_sc = ["#1976D2" if i != sc_pred else "#F44336" for i in range(n_classes)]
    ax2.bar(x, sc_norm, width, color=colors_sc, alpha=0.85)
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Normalized Score")
    ax2.set_title(f"SpikeCore (pred: {class_names[sc_pred]})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)

    match_str = "MATCH" if pt_pred == sc_pred else "MISMATCH"
    match_color = "green" if pt_pred == sc_pred else "red"
    fig.suptitle(f"{title}  [{match_str}]", fontsize=13, fontweight="bold", color=match_color)
    fig.tight_layout()
    return fig
