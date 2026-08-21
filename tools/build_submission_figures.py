from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_outputs" / "submission_figures_current"
SOURCE_DATA = OUTPUT / "source_data"

NAVY = "#18324A"
BLUE = "#3B6EA8"
TEAL = "#2A8C82"
ORANGE = "#D4863B"
RED = "#C45A4A"
GRAY = "#7A8793"
LIGHT_BLUE = "#EAF1F7"
LIGHT_TEAL = "#E8F4F1"
LIGHT_ORANGE = "#FAF0E5"
PANEL_BG = "#F7F9FB"
GRID = "#E5E9ED"
FULL_WIDTH_IN = 7.48  # Elsevier double-column width: approximately 190 mm.


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def rounded_box(ax, xy, width, height, text, fill, edge=NAVY, fontsize=7, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=fontsize,
        weight=weight,
        linespacing=1.25,
    )
    return box


def arrow(ax, start, end, color=GRAY, style="-|>", width=0.9):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=width,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def panel_label(ax, label: str, title: str | None = None) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")
    if title:
        ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=7.4, weight="bold", color=NAVY, va="bottom")


def pill(ax, xy, width, height, text, fill, edge="none", color=NAVY, fontsize=6.2, weight="bold"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={height / 2}",
        linewidth=0.7 if edge != "none" else 0,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", color=color, fontsize=fontsize, weight=weight)
    return patch


def draw_route_icon(ax, x0: float, y0: float, scale: float = 1.0) -> None:
    points = np.array(
        [
            [0.00, 0.15],
            [0.22, 0.15],
            [0.22, 0.48],
            [0.48, 0.48],
            [0.48, 0.72],
            [0.78, 0.72],
            [0.78, 0.92],
            [1.00, 0.92],
        ]
    )
    points[:, 0] = x0 + points[:, 0] * 0.16 * scale
    points[:, 1] = y0 + points[:, 1] * 0.13 * scale
    ax.plot(points[:, 0], points[:, 1], color=TEAL, linewidth=2.2, solid_capstyle="round", zorder=2)
    for index in (0, 2, 4, 6, 7):
        ax.add_patch(Circle(tuple(points[index]), 0.0065 * scale, facecolor="white", edgecolor=TEAL, linewidth=1.2, zorder=3))
    for fraction, color in ((0.34, BLUE), (0.59, ORANGE), (0.84, NAVY)):
        segment = int(fraction * (len(points) - 1))
        ax.add_patch(Circle(tuple(points[segment]), 0.008 * scale, facecolor=color, edgecolor="white", linewidth=0.6, zorder=4))


def build_framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.012, 0.965, "a", fontsize=9, weight="bold", va="top")
    ax.text(0.055, 0.952, "Offline decision-twin workflow", fontsize=8.5, weight="bold", color=NAVY, va="top")

    card_y, card_h = 0.43, 0.40
    cards = [
        (0.04, 0.25, LIGHT_BLUE, "Manufacturing-system twin"),
        (0.365, 0.27, LIGHT_TEAL, "Paired multistep world model"),
        (0.71, 0.25, LIGHT_ORANGE, "Selective decision support"),
    ]
    for x, width, fill, title in cards:
        rounded_box(ax, (x, card_y), width, card_h, "", fill)
        ax.text(x + 0.018, card_y + card_h - 0.045, title, fontsize=7.3, weight="bold", color=NAVY, va="top")

    draw_route_icon(ax, 0.075, 0.545, 1.05)
    ax.text(0.075, 0.525, "20-node route graph  |  3 AGVs", fontsize=6.1, color=NAVY, va="top")
    ax.text(0.075, 0.485, "Kinematics  ·  handling  ·  battery\ncharging  ·  path-resource contention", fontsize=5.8, color=GRAY, va="top", linespacing=1.35)

    pill(ax, (0.392, 0.675), 0.065, 0.055, "A0", LIGHT_BLUE, edge=BLUE, color=BLUE)
    pill(ax, (0.392, 0.585), 0.065, 0.055, "Ac", LIGHT_ORANGE, edge=ORANGE, color=ORANGE)
    arrow(ax, (0.458, 0.702), (0.535, 0.665), color=BLUE)
    arrow(ax, (0.458, 0.612), (0.535, 0.635), color=ORANGE)
    rounded_box(ax, (0.53, 0.60), 0.078, 0.13, "physics-graph\nbackbone", "white", fontsize=5.6, weight="bold")
    arrow(ax, (0.608, 0.665), (0.625, 0.665), color=TEAL)
    ax.text(0.54, 0.555, "paired effects", fontsize=5.6, color=NAVY, ha="left", weight="bold")
    for idx, label in enumerate(("120 s", "360 s", "720 s")):
        pill(ax, (0.475 + idx * 0.055, 0.465), 0.048, 0.047, label, "white", edge=TEAL, color=TEAL, fontsize=5.2)

    ax.text(0.738, 0.690, "rank candidates", fontsize=6.2, weight="bold", color=NAVY)
    ax.text(0.738, 0.642, "ensemble agreement + utility margin", fontsize=5.7, color=GRAY)
    ax.text(0.738, 0.594, "hard safety + cooldown checks", fontsize=5.7, color=GRAY)
    pill(ax, (0.742, 0.505), 0.177, 0.060, "shadow recommendation", "white", edge=TEAL, color=TEAL, fontsize=5.8)
    pill(ax, (0.742, 0.445), 0.177, 0.047, "fallback: retain DT-aware action", "#F9E8E4", color=RED, fontsize=5.25)

    arrow(ax, (0.29, 0.63), (0.365, 0.63), color=BLUE, width=1.4)
    arrow(ax, (0.635, 0.63), (0.71, 0.63), color=BLUE, width=1.4)

    ax.text(0.012, 0.315, "b", fontsize=9, weight="bold", va="top")
    ax.text(0.055, 0.305, "Evidence and authority ladder", fontsize=8.5, weight="bold", color=NAVY, va="top")
    stages = [
        (0.08, "1", "Physics factorial", "multistep fidelity"),
        (0.31, "2", "Unseen ranking", "counterfactual regret"),
        (0.54, "3", "Non-acting shadow", "coverage + benefit"),
        (0.77, "4", "Bounded authority", "abstain or fallback"),
    ]
    for x, number, title, subtitle in stages:
        ax.add_patch(Circle((x, 0.155), 0.022, facecolor=BLUE if number != "4" else ORANGE, edgecolor="white", linewidth=0.8, zorder=3))
        ax.text(x, 0.155, number, ha="center", va="center", color="white", fontsize=6.2, weight="bold", zorder=4)
        ax.text(x + 0.035, 0.174, title, fontsize=6.3, weight="bold", color=NAVY, va="center")
        ax.text(x + 0.035, 0.135, subtitle, fontsize=5.6, color=GRAY, va="center")
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax, (left[0] + 0.165, 0.155), (right[0] - 0.030, 0.155), color="#B7C2CC", width=1.0)

    save_figure(fig, OUTPUT / "figure_1_multistep_world_model_framework")
    plt.close(fig)


def build_cad_scene_figure() -> None:
    """Render the de-identified 20-node scene in a compact, publication-oriented layout."""
    scene_dir = ROOT / "agv-test2" / "simplified_cad_scenario"
    nodes_path = scene_dir / "simplified_nodes.csv"
    edges_path = scene_dir / "simplified_edges.csv"
    with nodes_path.open(encoding="utf-8-sig") as stream:
        nodes = list(csv.DictReader(stream))
    with edges_path.open(encoding="utf-8-sig") as stream:
        edges = list(csv.DictReader(stream))
    (SOURCE_DATA / "figure_2_nodes.csv").write_text(nodes_path.read_text(encoding="utf-8-sig"), encoding="utf-8")
    (SOURCE_DATA / "figure_2_edges.csv").write_text(edges_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    by_id = {int(row["node_id"]): row for row in nodes}

    def position(node_id: int) -> tuple[float, float]:
        row = by_id[node_id]
        # Rotate the de-identified coordinates by 90 degrees. Distances and topology are unchanged.
        return float(row["y_m"]), -float(row["x_m"])

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.25, 0.75), hspace=0.02)
    ax = fig.add_subplot(grid[0])
    ax_key = fig.add_subplot(grid[1])
    ax.set_aspect("equal")
    ax.set_xlim(-12, 78)
    ax.set_ylim(-47, 20)
    ax.axis("off")
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")

    zones = [
        (-10, 8, "Dispatch & energy", "#F4F7FA"),
        (8, 38, "Production services", "#FAF5EF"),
        (38, 58, "Shared corridor", "#F1F7F5"),
        (58, 77, "Warehouse interface", "#F2F5F8"),
    ]
    for left, right, label, color in zones:
        ax.add_patch(Rectangle((left, -45), right - left, 64, facecolor=color, edgecolor="none", zorder=0))
        ax.text(
            (left + right) / 2,
            17.6,
            label,
            ha="center",
            va="center",
            fontsize=5.6,
            weight="bold",
            color=GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        )

    for edge_row in edges:
        start = position(int(edge_row["from_node"]))
        end = position(int(edge_row["to_node"]))
        edge_type = edge_row["edge_type"]
        is_shared = edge_row["is_narrow_stress"] == "1"
        if edge_type.startswith("cad_trunk"):
            color, linewidth, linestyle = (ORANGE if is_shared else NAVY), 2.5, "-"
        elif edge_type == "warehouse_branch":
            color, linewidth, linestyle = GRAY, 1.4, "--"
        elif edge_type in {"home_lane", "charge_lane"}:
            color, linewidth, linestyle = BLUE, 1.4, "-"
        else:
            color, linewidth, linestyle = TEAL, 1.6, "-"
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, linestyle=linestyle, solid_capstyle="round", zorder=1)

    role_style = {
        "pickup": ("*", BLUE, 120),
        "guide": ("o", NAVY, 34),
        "control": ("o", "#F2A900", 34),
        "home": ("o", "#5B9E55", 45),
        "charge": ("D", "#22AFC2", 60),
        "workstation": ("^", "#8F5AA2", 62),
        "buffer": ("D", ORANGE, 55),
        "warehouse": ("s", "#4A9D45", 70),
        "warehouse_slot": ("s", BLUE, 55),
    }
    for row in nodes:
        x, y = position(int(row["node_id"]))
        marker, color, size = role_style[row["role"]]
        ax.scatter(x, y, marker=marker, s=size, color=color, edgecolor="white", linewidth=0.7, zorder=4)

    label_offsets = {
        0: (1.0, 1.5, "left"),
        8: (1.2, -1.6, "left"),
        9: (0.0, 1.6, "center"),
        10: (0.0, 1.6, "center"),
        11: (0.0, 1.6, "center"),
        12: (-1.0, -1.6, "right"),
        13: (0.0, 1.6, "center"),
        14: (0.0, 1.6, "center"),
        15: (0.0, 1.6, "center"),
        16: (1.1, 0.0, "left"),
        17: (0.0, 1.6, "center"),
        18: (-1.0, 0.0, "right"),
        19: (-1.0, 0.0, "right"),
    }
    for node_id, (dx, dy, alignment) in label_offsets.items():
        x, y = position(node_id)
        ax.text(
            x + dx,
            y + dy,
            str(node_id),
            ha=alignment,
            va="center",
            fontsize=5.0,
            weight="bold",
            color=NAVY,
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.65},
        )

    for node_id in range(1, 8):
        x, y = position(node_id)
        guide_label = by_id[node_id]["node_name"].replace("G2_G3_Mid", "G2-G3 mid")
        ax.text(
            x,
            y - 1.45,
            guide_label,
            ha="center",
            va="top",
            fontsize=4.6,
            color=GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.45},
        )

    # Primary material-flow direction and a physical scale bar.
    arrow(ax, (14, -6.4), (24, -6.4), color=NAVY, width=1.0)
    ax.text(
        19,
        -8.0,
        "principal material flow",
        ha="center",
        fontsize=4.8,
        color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.8},
    )
    ax.plot([58, 68], [-39.0, -39.0], color=NAVY, linewidth=1.5)
    ax.plot([58, 58], [-39.7, -38.3], color=NAVY, linewidth=0.8)
    ax.plot([68, 68], [-39.7, -38.3], color=NAVY, linewidth=0.8)
    ax.text(63, -37.2, "10 m", ha="center", fontsize=5.1, color=NAVY)

    legend_handles = [
        Line2D([0], [0], color=NAVY, linewidth=2.4, label="Main corridor"),
        Line2D([0], [0], color=ORANGE, linewidth=2.4, label="Shared / stress-sensitive path"),
        Line2D([0], [0], color=TEAL, linewidth=1.6, label="Service branch"),
        Line2D([0], [0], color=GRAY, linewidth=1.4, linestyle="--", label="Warehouse branch"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        fontsize=5.1,
        handlelength=2.5,
        columnspacing=1.25,
        frameon=False,
    )

    # Keep full functional names outside the route drawing to prevent label collisions.
    ax_key.add_patch(
        FancyBboxPatch(
            (0.012, 0.08),
            0.976,
            0.84,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=ax_key.transAxes,
            facecolor="#F8FAFC",
            edgecolor="#D6DEE6",
            linewidth=0.7,
        )
    )
    ax_key.text(0.028, 0.82, "Functional nodes (IDs match map)", fontsize=5.8, weight="bold", color=NAVY, va="center")
    key_items = [
        (0, "Pickup A", "pickup"),
        (8, "Warehouse B", "warehouse"),
        ("9–11", "AGV homes", "home"),
        (12, "Charging", "charge"),
        (13, "Packaging", "workstation"),
        (14, "Labeling", "workstation"),
        (15, "Prep buffer", "buffer"),
        (16, "Passing bay", "buffer"),
        (17, "Material buffer", "buffer"),
        ("18–19", "Storage W1 / W2", "warehouse_slot"),
    ]
    for index, (node_id, label, role) in enumerate(key_items):
        row_index, column_index = divmod(index, 5)
        x = 0.035 + column_index * 0.193
        y = 0.56 - row_index * 0.31
        marker, color, _ = role_style[role]
        ax_key.scatter(x, y, marker=marker, s=27, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax_key.text(x + 0.018, y, f"{node_id}  {label}", fontsize=5.1, color=NAVY, va="center", ha="left")

    save_figure(fig, OUTPUT / "figure_2_cad_derived_scene")
    plt.close(fig)


def build_detailed_model_figure() -> None:
    """Explain the frozen multistep backbone and trainable paired-effect head."""
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.05))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.05, 1.0, 0.68), hspace=0.25)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    ax_c = fig.add_subplot(grid[2])
    for axis in (ax_a, ax_b, ax_c):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    ax_a.text(0.0, 0.98, "a", fontsize=9, weight="bold", va="top")
    ax_a.text(0.045, 0.98, "Physics-aware graph-state backbone", fontsize=8.2, weight="bold", color=NAVY, va="top")
    rounded_box(ax_a, (0.03, 0.18), 0.23, 0.58, "", LIGHT_BLUE)
    ax_a.text(0.05, 0.69, "Graph-state inputs", fontsize=6.8, weight="bold", color=NAVY)
    draw_route_icon(ax_a, 0.062, 0.43, 0.92)
    pill(ax_a, (0.05, 0.32), 0.19, 0.055, "dynamic: occupancy · AGV · SOC · jobs", "white", color=BLUE, fontsize=5.1)
    pill(ax_a, (0.05, 0.245), 0.19, 0.055, "static: geometry · distance · capacity", "white", color=TEAL, fontsize=5.1)

    rounded_box(ax_a, (0.325, 0.25), 0.18, 0.44, "", LIGHT_TEAL)
    ax_a.text(0.415, 0.61, "Edge-conditioned\ngraph attention", ha="center", va="center", fontsize=6.5, weight="bold", color=NAVY)
    graph_nodes = [(0.37, 0.40), (0.42, 0.48), (0.46, 0.37), (0.45, 0.55)]
    for left, right in ((0, 1), (1, 2), (1, 3), (2, 3)):
        ax_a.plot([graph_nodes[left][0], graph_nodes[right][0]], [graph_nodes[left][1], graph_nodes[right][1]], color="#8EB6AE", linewidth=1.1)
    for index, point in enumerate(graph_nodes):
        ax_a.add_patch(Circle(point, 0.012, facecolor=TEAL if index == 1 else "white", edgecolor=TEAL, linewidth=1.1, zorder=3))
    ax_a.text(0.415, 0.205, "node/AGV encoders + physical edge bias", ha="center", fontsize=5.2, color=GRAY)

    ax_a.text(0.665, 0.69, "Residual multistep rollout", fontsize=6.8, weight="bold", color=NAVY, ha="center")
    for index, (dx, label) in enumerate(((0.00, "t+1"), (0.045, "t+5"), (0.09, "t+10"))):
        rect = FancyBboxPatch((0.575 + dx, 0.31 + index * 0.025), 0.16, 0.26, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=[LIGHT_BLUE, LIGHT_TEAL, "#F3F5F7"][index], edgecolor=NAVY, linewidth=0.8, zorder=2 + index)
        ax_a.add_patch(rect)
        ax_a.text(0.595 + dx, 0.51 + index * 0.025, label, fontsize=5.6, weight="bold", color=NAVY, zorder=4)
    ax_a.text(0.695, 0.205, "shared transition weights", fontsize=5.2, color=GRAY, ha="center")

    rounded_box(ax_a, (0.82, 0.22), 0.15, 0.53, "", LIGHT_ORANGE)
    ax_a.text(0.895, 0.67, "Prediction heads", fontsize=6.7, weight="bold", color=NAVY, ha="center")
    for y, label, color in ((0.57, "AGV / node state", BLUE), (0.48, "time + energy", TEAL), (0.39, "tasks + queue", ORANGE), (0.30, "charge risk", RED)):
        pill(ax_a, (0.84, y), 0.11, 0.055, label, "white", color=color, fontsize=5.2)
    for start, end in (((0.26, 0.48), (0.325, 0.48)), ((0.505, 0.48), (0.575, 0.48)), ((0.755, 0.48), (0.82, 0.48))):
        arrow(ax_a, start, end, color=BLUE, width=1.1)

    ax_b.text(0.0, 0.98, "b", fontsize=9, weight="bold", va="top")
    ax_b.text(0.045, 0.98, "Frozen paired counterfactual inference", fontsize=8.2, weight="bold", color=NAVY, va="top")
    rounded_box(ax_b, (0.03, 0.33), 0.14, 0.38, "State S(t)\n+ frozen arrivals", "#F3F5F7", fontsize=6.2, weight="bold")
    pill(ax_b, (0.215, 0.62), 0.12, 0.10, "baseline action  A0", LIGHT_BLUE, edge=BLUE, color=BLUE, fontsize=5.8)
    pill(ax_b, (0.215, 0.32), 0.12, 0.10, "candidate action  Ac", LIGHT_ORANGE, edge=ORANGE, color=ORANGE, fontsize=5.8)
    rounded_box(ax_b, (0.39, 0.37), 0.20, 0.35, "Frozen V13 backbone\nshared weights, applied twice\n336,748 total parameters", LIGHT_TEAL, fontsize=6.0, weight="bold")
    ax_b.add_patch(Circle((0.645, 0.545), 0.035, facecolor=NAVY, edgecolor="white", linewidth=0.8))
    ax_b.text(0.645, 0.545, "Δ", color="white", fontsize=8, weight="bold", ha="center", va="center")
    rounded_box(ax_b, (0.705, 0.37), 0.16, 0.35, "Trainable paired head\nshared MLP\n56,457 parameters", LIGHT_ORANGE, fontsize=6.0, weight="bold")
    ax_b.text(0.925, 0.73, "paired effects", fontsize=6.5, weight="bold", color=NAVY, ha="center")
    for y, label in ((0.61, "120 s"), (0.51, "360 s"), (0.41, "720 s")):
        pill(ax_b, (0.885, y), 0.08, 0.065, label, "white", edge=TEAL, color=TEAL, fontsize=5.5)
    ax_b.text(0.925, 0.31, "energy · tasks · charge queue", fontsize=5.2, color=GRAY, ha="center")
    arrow(ax_b, (0.17, 0.52), (0.215, 0.67), color=BLUE)
    arrow(ax_b, (0.17, 0.52), (0.215, 0.37), color=ORANGE)
    arrow(ax_b, (0.335, 0.67), (0.39, 0.61), color=BLUE)
    arrow(ax_b, (0.335, 0.37), (0.39, 0.47), color=ORANGE)
    arrow(ax_b, (0.59, 0.545), (0.61, 0.545), color=GRAY)
    arrow(ax_b, (0.68, 0.545), (0.705, 0.545), color=GRAY)
    arrow(ax_b, (0.865, 0.545), (0.885, 0.545), color=TEAL)

    ax_c.text(0.0, 0.98, "c", fontsize=9, weight="bold", va="top")
    ax_c.text(0.045, 0.98, "Selective authority gate", fontsize=8.2, weight="bold", color=NAVY, va="top")
    stages = [
        (0.05, 0.18, "Rank candidates", "normalized utility", LIGHT_BLUE),
        (0.29, 0.18, "Agreement gate", "3 seeds + margin", LIGHT_TEAL),
        (0.53, 0.18, "Safety gate", "hard rules + budget", "#F3F5F7"),
        (0.77, 0.18, "Shadow advice", "recommend or abstain", LIGHT_ORANGE),
    ]
    for x, width, title, subtitle, fill in stages:
        rounded_box(ax_c, (x, 0.31), width, 0.34, "", fill)
        ax_c.text(x + width / 2, 0.53, title, ha="center", fontsize=6.5, weight="bold", color=NAVY)
        ax_c.text(x + width / 2, 0.41, subtitle, ha="center", fontsize=5.5, color=GRAY)
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax_c, (left[0] + left[1], 0.48), (right[0], 0.48), color=BLUE, width=1.1)
    pill(ax_c, (0.77, 0.20), 0.18, 0.07, "fallback: retain DT-aware action", "#F9E8E4", color=RED, fontsize=5.4)

    save_figure(fig, OUTPUT / "figure_3_detailed_world_model_architecture")
    plt.close(fig)


def build_detailed_model_figure_jms() -> None:
    """JMS-oriented architecture diagram with explicit paired branches."""
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.05))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.08, 0.66), hspace=0.25)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    ax_c = fig.add_subplot(grid[2])
    for axis in (ax_a, ax_b, ax_c):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    ax_a.text(0.0, 0.98, "a", fontsize=9, weight="bold", color=NAVY, va="top")
    ax_a.text(0.045, 0.98, "Physics-aware graph-state backbone", fontsize=8.2, weight="bold", color=NAVY, va="top")
    rounded_box(ax_a, (0.03, 0.18), 0.22, 0.58, "", LIGHT_BLUE)
    ax_a.text(0.14, 0.69, "Graph-state inputs", fontsize=6.8, weight="bold", color=NAVY, ha="center")
    draw_route_icon(ax_a, 0.058, 0.43, 0.92)
    ax_a.text(0.14, 0.33, "dynamic: occupancy | AGV | SOC | jobs", ha="center", fontsize=5.1, color=BLUE, weight="bold")
    ax_a.text(0.14, 0.25, "static: geometry | distance | capacity", ha="center", fontsize=5.1, color=TEAL, weight="bold")

    rounded_box(ax_a, (0.31, 0.25), 0.19, 0.44, "", LIGHT_TEAL, edge=TEAL)
    ax_a.text(0.405, 0.61, "Edge-conditioned\ngraph attention", ha="center", va="center", fontsize=6.5, weight="bold", color=NAVY)
    graph_nodes = [(0.36, 0.40), (0.41, 0.48), (0.45, 0.37), (0.44, 0.55)]
    for left, right in ((0, 1), (1, 2), (1, 3), (2, 3)):
        ax_a.plot([graph_nodes[left][0], graph_nodes[right][0]], [graph_nodes[left][1], graph_nodes[right][1]], color="#8EB6AE", linewidth=1.1)
    for index, point in enumerate(graph_nodes):
        ax_a.add_patch(Circle(point, 0.012, facecolor=TEAL if index == 1 else "white", edgecolor=TEAL, linewidth=1.1, zorder=3))
    ax_a.text(0.405, 0.16, "node/AGV encoders + physical edge bias", ha="center", fontsize=5.2, color=GRAY)

    rounded_box(ax_a, (0.56, 0.24), 0.24, 0.48, "", LIGHT_ORANGE, edge=ORANGE)
    ax_a.text(0.68, 0.64, "Residual multistep rollout", fontsize=6.8, weight="bold", color=NAVY, ha="center")
    for x, label in ((0.58, "t+1"), (0.655, "t+5"), (0.73, "t+10")):
        pill(ax_a, (x, 0.40), 0.055, 0.10, label, "white", edge=ORANGE, color=ORANGE, fontsize=5.5)
    ax_a.text(0.68, 0.30, "shared residual transition weights", fontsize=5.2, color=GRAY, ha="center")

    rounded_box(ax_a, (0.84, 0.20), 0.13, 0.55, "", "#F3F5F7", edge=GRAY)
    ax_a.text(0.905, 0.67, "Prediction heads", fontsize=6.7, weight="bold", color=NAVY, ha="center")
    for y, label, color in ((0.56, "state transition", BLUE), (0.47, "time + energy", TEAL), (0.38, "tasks + queue", ORANGE), (0.29, "charge risk", RED)):
        pill(ax_a, (0.855, y), 0.10, 0.055, label, "white", color=color, fontsize=5.1)
    for start, end in (((0.25, 0.48), (0.31, 0.48)), ((0.50, 0.48), (0.56, 0.48)), ((0.80, 0.48), (0.84, 0.48))):
        arrow(ax_a, start, end, color=BLUE, width=1.1)

    ax_b.text(0.0, 0.98, "b", fontsize=9, weight="bold", color=NAVY, va="top")
    ax_b.text(0.045, 0.98, "Frozen paired counterfactual inference", fontsize=8.2, weight="bold", color=NAVY, va="top")
    rounded_box(ax_b, (0.03, 0.30), 0.15, 0.42, "State S(t)\n+ frozen arrivals", "#F3F5F7", fontsize=6.2, weight="bold")
    pill(ax_b, (0.22, 0.63), 0.14, 0.10, "baseline action A0", LIGHT_BLUE, edge=BLUE, color=BLUE, fontsize=5.7)
    pill(ax_b, (0.22, 0.27), 0.14, 0.10, "candidate action Ac", LIGHT_ORANGE, edge=ORANGE, color=ORANGE, fontsize=5.7)
    rounded_box(ax_b, (0.41, 0.58), 0.19, 0.17, "Frozen V13\nphysics-graph backbone", LIGHT_TEAL, edge=TEAL, fontsize=5.8, weight="bold")
    rounded_box(ax_b, (0.41, 0.24), 0.19, 0.17, "Frozen V13\nphysics-graph backbone", LIGHT_TEAL, edge=TEAL, fontsize=5.8, weight="bold")
    ax_b.text(0.505, 0.49, "shared frozen weights | 336,748 parameters", fontsize=5.2, color=TEAL, weight="bold", ha="center")
    rounded_box(ax_b, (0.66, 0.35), 0.13, 0.27, "Paired effect\n(Ac - A0)", LIGHT_ORANGE, edge=ORANGE, fontsize=6.0, weight="bold")
    rounded_box(ax_b, (0.84, 0.34), 0.13, 0.29, "Trainable paired head\nshared MLP\n56,457 parameters", "#F9E8E4", edge=RED, fontsize=5.7, weight="bold")
    ax_b.text(0.815, 0.24, "fixed physical horizons", fontsize=5.2, color=GRAY, ha="center")
    for x, label in ((0.67, "120 s"), (0.77, "360 s"), (0.87, "720 s")):
        pill(ax_b, (x, 0.13), 0.075, 0.065, label, "white", edge=TEAL, color=TEAL, fontsize=5.3)
    ax_b.text(0.815, 0.07, "energy | tasks | charge queue", fontsize=5.2, color=GRAY, ha="center")
    arrow(ax_b, (0.18, 0.51), (0.22, 0.68), color=BLUE)
    arrow(ax_b, (0.18, 0.51), (0.22, 0.32), color=ORANGE)
    arrow(ax_b, (0.36, 0.68), (0.41, 0.665), color=BLUE)
    arrow(ax_b, (0.36, 0.32), (0.41, 0.325), color=ORANGE)
    arrow(ax_b, (0.60, 0.665), (0.66, 0.54), color=BLUE)
    arrow(ax_b, (0.60, 0.325), (0.66, 0.43), color=TEAL)
    arrow(ax_b, (0.79, 0.485), (0.84, 0.485), color=ORANGE)

    ax_c.text(0.0, 0.98, "c", fontsize=9, weight="bold", color=NAVY, va="top")
    ax_c.text(0.045, 0.98, "Selective authority gate", fontsize=8.2, weight="bold", color=NAVY, va="top")
    stages = [
        (0.05, 0.18, "Rank candidates", "normalized utility", LIGHT_BLUE),
        (0.29, 0.18, "Agreement gate", "3 seeds + utility margin", LIGHT_TEAL),
        (0.53, 0.18, "Safety gate", "hard rules + cooldown", "#F3F5F7"),
        (0.77, 0.18, "Shadow advice", "recommend / abstain / fallback", LIGHT_ORANGE),
    ]
    for x, width, title, subtitle, fill in stages:
        rounded_box(ax_c, (x, 0.31), width, 0.34, "", fill)
        ax_c.text(x + width / 2, 0.53, title, ha="center", fontsize=6.5, weight="bold", color=NAVY)
        ax_c.text(x + width / 2, 0.41, subtitle, ha="center", fontsize=5.5, color=GRAY)
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax_c, (left[0] + left[1], 0.48), (right[0], 0.48), color=BLUE, width=1.1)
    save_figure(fig, OUTPUT / "figure_3_detailed_world_model_architecture")
    plt.close(fig)


def build_decision_evidence_figure() -> None:
    ranking_path = (
        ROOT
        / "experiment_results"
        / "world_model_counterfactual_v144_ranking_confirmation_v1"
        / "V144_RANKING_CONFIRMATION_AUDIT.json"
    )
    shadow_path = (
        ROOT
        / "experiment_results"
        / "world_model_counterfactual_v145_shadow_confirmation_parallel_v2"
        / "V145_SHADOW_CONFIRMATION_AUDIT.json"
    )
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    rank_rows = ranking["episode_results"]
    shadow_rows = shadow["episode_results"]

    with (SOURCE_DATA / "figure_5_ranking_by_trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)
    with (SOURCE_DATA / "figure_5_shadow_by_trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(shadow_rows[0]))
        writer.writeheader()
        writer.writerows(shadow_rows)

    episode = np.arange(1, len(rank_rows) + 1)
    model_regret = np.array([row["model_mean_regret"] for row in rank_rows])
    baseline_regret = np.array([row["baseline_mean_regret"] for row in rank_rows])
    reduction = np.array([row["regret_reduction"] for row in rank_rows])
    coverage = np.array([row["coverage"] for row in shadow_rows])
    precision = np.array([row["benefit_precision"] for row in shadow_rows])
    true_gain = np.array([row["mean_true_gain"] for row in shadow_rows])

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.05))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.22, 1.0), hspace=0.46, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    for x, b, m in zip(episode, baseline_regret, model_regret):
        color = TEAL if m < b else RED
        ax_a.plot([x, x], [b, m], color=color, linewidth=1.2, alpha=0.75)
    ax_a.scatter(episode, baseline_regret, s=18, color=GRAY, label="DT-aware candidate", zorder=3)
    ax_a.scatter(episode, model_regret, s=20, color=TEAL, label="Frozen model ensemble", zorder=3)
    ax_a.set_xlabel("Unseen trajectory")
    ax_a.set_ylabel("Mean normalized ranking regret")
    ax_a.set_xticks(episode)
    ax_a.legend(loc="upper left")
    ax_a.text(
        0.98,
        0.86,
        "11/12 lower regret",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        color=TEAL,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    ax_a.grid(axis="y", color=GRID, linewidth=0.6)

    bars = ax_b.bar(episode, reduction * 100, color=np.where(reduction >= 0, TEAL, RED), width=0.72)
    ax_b.axhline(0, color=NAVY, linewidth=0.7)
    ax_b.axhline(ranking["episode_bootstrap"]["mean"] * 100, color=BLUE, linewidth=1.0, linestyle="--")
    ax_b.set_xlabel("Unseen trajectory")
    ax_b.set_ylabel("Regret reduction (%)")
    ax_b.set_xticks([1, 4, 8, 12])
    ax_b.text(
        0.97,
        0.96,
        "Mean 34.7%\n95% CI 19.8-50.6%",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        color=BLUE,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
    )
    ax_b.grid(axis="y", color=GRID, linewidth=0.6)

    width = 0.36
    ax_c.bar(episode - width / 2, coverage * 100, width, color=BLUE, label="Recommendation coverage")
    ax_c.bar(episode + width / 2, precision * 100, width, color=ORANGE, label="Positive-utility fraction")
    ax_c.axhline(shadow["shadow_summary"]["coverage"] * 100, color=BLUE, linestyle="--", linewidth=0.8)
    ax_c.axhline(shadow["shadow_summary"]["benefit_precision"] * 100, color=ORANGE, linestyle="--", linewidth=0.8)
    ax_c.set_xlabel("Independent shadow trajectory")
    ax_c.set_ylabel("Rate (%)")
    ax_c.set_ylim(0, 105)
    ax_c.set_xticks(episode)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, borderaxespad=0, fontsize=5.8, handlelength=1.5)
    ax_c.grid(axis="y", color=GRID, linewidth=0.6)

    ax_d.bar(episode, true_gain, color=TEAL, width=0.72)
    ax_d.axhline(0, color=NAVY, linewidth=0.7)
    ax_d.axhline(shadow["shadow_summary"]["mean_true_gain"], color=BLUE, linewidth=1.0, linestyle="--")
    ci_low, ci_high = shadow["trajectory_bootstrap"]["mean_gain_ci95"]
    ax_d.fill_between([0.5, 12.5], ci_low, ci_high, color=BLUE, alpha=0.10, linewidth=0)
    ax_d.set_xlabel("Independent shadow trajectory")
    ax_d.set_ylabel("Mean realized normalized utility")
    ax_d.set_xticks([1, 4, 8, 12])
    ax_d.set_ylim(0, 0.90)
    ax_d.text(
        0.97,
        0.96,
        "All 12 trajectories positive\nMean 0.351; 95% CI 0.242-0.465",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        color=TEAL,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
    )
    ax_d.grid(axis="y", color=GRID, linewidth=0.6)

    panel_label(ax_a, "a", "Paired ranking regret on unseen trajectories")
    panel_label(ax_b, "b", "Trajectory-level regret reduction")
    panel_label(ax_c, "c", "Selective shadow coverage and benefit precision")
    panel_label(ax_d, "d", "Realized utility of shadow recommendations")
    save_figure(fig, OUTPUT / "figure_5_counterfactual_decision_evidence")
    plt.close(fig)


def build_physics_factorial_figure() -> None:
    evidence_dir = ROOT / "experiment_results" / "v11_physics_factorial_arrival_v4_independent_v2"
    with (evidence_dir / "condition_summary.csv").open(encoding="utf-8-sig") as stream:
        condition_rows = list(csv.DictReader(stream))
    with (evidence_dir / "paired_episode_bootstrap.csv").open(encoding="utf-8-sig") as stream:
        bootstrap_rows = list(csv.DictReader(stream))

    composite = [
        row
        for row in condition_rows
        if row["horizon_steps"] == "5_and_10" and row["metric"] == "normalized_composite"
    ]
    order = ["Full", "No physics loss", "No physical features", "Data-only graph"]
    by_condition = {row["condition"]: row for row in composite}
    means = np.array([float(by_condition[name]["mae"]) for name in order])
    spreads = np.array([float(by_condition[name]["episode_sd"]) for name in order])

    comparisons = [
        "Full minus Data-only graph composite",
        "Physics-loss main effect (on minus off)",
        "Physical-feature main effect (on minus off)",
    ]
    comparison_labels = ["Full vs data-only", "Physics-loss effect", "Physical-feature effect"]
    by_comparison = {row["comparison"]: row for row in bootstrap_rows}
    effects = np.array([float(by_comparison[name]["delta_mean"]) for name in comparisons])
    low = np.array([float(by_comparison[name]["ci_low"]) for name in comparisons])
    high = np.array([float(by_comparison[name]["ci_high"]) for name in comparisons])

    with (SOURCE_DATA / "figure_4_physics_factorial_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "normalized_composite_mae", "episode_sd", "episode_count"])
        for name in order:
            row = by_condition[name]
            writer.writerow([name, row["mae"], row["episode_sd"], row["episode_count"]])
    with (SOURCE_DATA / "figure_4_physics_factorial_effects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "delta_mean", "ci_low", "ci_high", "episode_count"])
        for label, name in zip(comparison_labels, comparisons):
            row = by_comparison[name]
            writer.writerow([label, row["delta_mean"], row["ci_low"], row["ci_high"], row["episode_count"]])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.85), gridspec_kw={"width_ratios": [1.02, 1.0]})
    colors = [TEAL, "#80B9AE", "#A7B3BE", GRAY]
    x = np.arange(len(order))
    ax_a.bar(x, means, yerr=spreads, capsize=3, color=colors, width=0.70, ecolor=NAVY, linewidth=0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Full", "No physics\nloss", "No physical\nfeatures", "Data-only\ngraph"])
    ax_a.set_ylabel("Normalized 5+10-step composite MAE")
    ax_a.set_ylim(0, 1.34)
    bracket_y = 1.245
    ax_a.plot([0, 0, 3, 3], [bracket_y - 0.025, bracket_y, bracket_y, bracket_y - 0.025], color=TEAL, linewidth=0.9, clip_on=False)
    ax_a.text(1.5, bracket_y + 0.025, "Full 4.26% lower than data-only", ha="center", va="bottom", color=TEAL, weight="bold", fontsize=6.2)
    ax_a.grid(axis="y", color=GRID, linewidth=0.6)

    y = np.arange(len(comparison_labels))[::-1]
    ax_b.axvspan(-0.085, 0, color=LIGHT_TEAL, alpha=0.65, zorder=0)
    ax_b.axvline(0, color=NAVY, linewidth=0.8)
    for yi, effect, lo, hi in zip(y, effects, low, high):
        significant = hi < 0 or lo > 0
        color = TEAL if significant else GRAY
        ax_b.errorbar(
            effect,
            yi,
            xerr=[[effect - lo], [hi - effect]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            markersize=5,
        )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(comparison_labels)
    ax_b.set_xlabel("Paired change in normalized composite MAE")
    ax_b.set_xlim(-0.085, 0.015)
    ax_b.text(0.03, 0.04, "improvement  ←", transform=ax_b.transAxes, va="bottom", color=TEAL, weight="bold", fontsize=6.0)
    ax_b.grid(axis="x", color=GRID, linewidth=0.6)

    panel_label(ax_a, "a", "Prediction error by factorial condition")
    panel_label(ax_b, "b", "Paired effects with trajectory-bootstrap 95% CI")
    fig.tight_layout()
    save_figure(fig, OUTPUT / "figure_4_physics_factorial_evidence")
    plt.close(fig)


def build_paired_formulation_figure() -> None:
    evidence_path = (
        ROOT
        / "experiment_results"
        / "v151_paired_vs_absolute_confirmation_seed18400"
        / "V151_PAIRED_FORMULATION_AUDIT.json"
    )
    sensitivity_path = (
        ROOT
        / "experiment_results"
        / "v151_utility_sensitivity_v1"
        / "utility_sensitivity_summary.csv"
    )
    audit = json.loads(evidence_path.read_text(encoding="utf-8"))
    episode_rows = audit["episode_results"]
    paired = audit["paired_episode_bootstrap"]
    with sensitivity_path.open(encoding="utf-8-sig") as stream:
        sensitivity_rows = list(csv.DictReader(stream))

    with (SOURCE_DATA / "figure_6_paired_formulation_by_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    with (SOURCE_DATA / "figure_6_paired_formulation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "mean_regret", "regret_reduction", "top1_agreement", "trainable_parameters"])
        writer.writerow(
            [
                "Direct paired-effect",
                audit["direct_delta_ranking"]["model_mean_regret"],
                audit["direct_delta_ranking"]["regret_reduction"],
                audit["direct_delta_ranking"]["top1_agreement"],
                56457,
            ]
        )
        writer.writerow(
            [
                "Absolute then difference",
                audit["absolute_outcome_ranking"]["model_mean_regret"],
                audit["absolute_outcome_ranking"]["regret_reduction"],
                audit["absolute_outcome_ranking"]["top1_agreement"],
                56457,
            ]
        )
        writer.writerow(["DT-aware unchanged", audit["direct_delta_ranking"]["baseline_mean_regret"], 0.0, "", ""])
        writer.writerow(["Random choice", "", "", audit["direct_delta_ranking"]["random_top1"], ""])
        writer.writerow(["Absolute minus direct", paired["mean"], "", "", ""])
        writer.writerow(["95% CI low", paired["ci_low"], "", "", ""])
        writer.writerow(["95% CI high", paired["ci_high"], "", "", ""])
    with (SOURCE_DATA / "figure_6_paired_formulation_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    regrets = [
        audit["direct_delta_ranking"]["model_mean_regret"],
        audit["absolute_outcome_ranking"]["model_mean_regret"],
        audit["direct_delta_ranking"]["baseline_mean_regret"],
    ]
    top1 = [
        audit["direct_delta_ranking"]["top1_agreement"],
        audit["absolute_outcome_ranking"]["top1_agreement"],
        audit["direct_delta_ranking"]["random_top1"],
    ]
    differences = np.array([row["absolute_minus_direct_regret"] for row in episode_rows])
    episodes = np.arange(1, len(differences) + 1)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.85))
    grid = fig.add_gridspec(2, 3, height_ratios=(0.92, 1.08), hspace=0.53, wspace=0.50)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0:2])
    ax_e = fig.add_subplot(grid[1, 2])

    method_colors = [TEAL, ORANGE, GRAY]
    method_names = ["Direct paired\neffect", "Absolute then\ndifference", "DT-aware\nunchanged"]
    y = np.arange(3)[::-1]
    ax_a.barh(y, regrets, color=method_colors, height=0.60)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(method_names)
    ax_a.set_xlabel("Mean normalized ranking regret")
    ax_a.set_xlim(0, 0.59)
    for yi, value in zip(y, regrets):
        ax_a.text(value + 0.010, yi, f"{value:.3f}", va="center", fontsize=6.3, color=NAVY)
    ax_a.text(0.98, 0.96, "lower is better", transform=ax_a.transAxes, ha="right", va="top", color=GRAY, fontsize=5.6)
    ax_a.grid(axis="x", color=GRID, linewidth=0.6)

    ax_b.bar(np.arange(3), np.array(top1) * 100, color=method_colors, width=0.64)
    ax_b.set_xticks(np.arange(3))
    ax_b.set_xticklabels(["Direct", "Absolute", "Random"])
    ax_b.set_ylabel("Top-1 agreement (%)")
    ax_b.set_ylim(0, 48)
    for x, value in enumerate(top1):
        ax_b.text(x, value * 100 + 1.2, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=6.2, color=NAVY)
    ax_b.text(0.98, 0.96, "higher is better", transform=ax_b.transAxes, ha="right", va="top", color=GRAY, fontsize=5.6)
    ax_b.grid(axis="y", color=GRID, linewidth=0.6)

    mean = paired["mean"]
    low = paired["ci_low"]
    high = paired["ci_high"]
    ax_d.axvline(0, color=NAVY, linewidth=0.8)
    ax_d.errorbar(
        mean,
        0.64,
        xerr=[[mean - low], [high - mean]],
        fmt="o",
        color=GRAY,
        ecolor=GRAY,
        capsize=4,
        markersize=6,
        elinewidth=1.5,
    )
    ax_d.set_xlim(-0.045, 0.255)
    ax_d.set_ylim(0, 1)
    ax_d.set_yticks([])
    ax_d.set_xlabel("Absolute-minus-direct regret")
    ax_d.grid(axis="x", color=GRID, linewidth=0.6)
    ax_d.text(
        0.5,
        0.35,
        f"Mean {mean:+.3f}\n95% CI [{low:+.3f}, {high:+.3f}]",
        transform=ax_d.transAxes,
        ha="center",
        va="center",
        color=GRAY,
        weight="bold",
        fontsize=5.8,
        linespacing=1.35,
    )
    ax_d.text(
        0.5,
        0.08,
        "CI crosses zero: not conclusive",
        transform=ax_d.transAxes,
        ha="center",
        va="center",
        fontsize=5.2,
        color=RED,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F9E8E4", "edgecolor": "none"},
    )

    ax_c.bar(episodes, differences, color=np.where(differences > 0, TEAL, ORANGE), width=0.72)
    ax_c.axhline(0, color=NAVY, linewidth=0.8)
    ax_c.set_xticks(episodes)
    ax_c.set_xlabel("Independent confirmation trajectory")
    ax_c.set_ylabel("Absolute-minus-direct regret")
    ax_c.set_ylim(-0.31, 0.44)
    ax_c.text(0.6, 0.405, "teal: direct lower", ha="left", va="top", color=TEAL, weight="bold", fontsize=5.2)
    ax_c.text(0.6, -0.285, "orange: absolute lower", ha="left", va="bottom", color=ORANGE, weight="bold", fontsize=5.2)
    ax_c.text(12.4, 0.405, "8/12 favor direct", ha="right", va="top", color=NAVY, weight="bold", fontsize=5.5)
    ax_c.grid(axis="y", color=GRID, linewidth=0.6)

    sensitivity_labels = ["Equal\n1:1:1", "Energy\n2:1:1", "Throughput\n1:2:1", "Queue\n1:1:2"]
    sensitivity_effects = np.array([float(row["absolute_minus_direct_regret"]) for row in sensitivity_rows])
    sensitivity_low = np.array([float(row["difference_ci_low"]) for row in sensitivity_rows])
    sensitivity_high = np.array([float(row["difference_ci_high"]) for row in sensitivity_rows])
    sensitivity_y = np.arange(len(sensitivity_rows))[::-1]
    ax_e.axvline(0, color=NAVY, linewidth=0.8)
    for yi, effect, lo, hi in zip(sensitivity_y, sensitivity_effects, sensitivity_low, sensitivity_high):
        supported = lo > 0
        color = TEAL if supported else GRAY
        ax_e.errorbar(
            effect,
            yi,
            xerr=[[effect - lo], [hi - effect]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=4.8,
        )
    ax_e.set_yticks(sensitivity_y)
    ax_e.set_yticklabels(sensitivity_labels)
    ax_e.set_xlabel("Absolute-minus-direct regret")
    ax_e.set_xlim(-0.075, 0.51)
    ax_e.grid(axis="x", color=GRID, linewidth=0.6)

    panel_label(ax_a, "a", "Ranking regret")
    panel_label(ax_b, "b", "Top-1 action agreement")
    panel_label(ax_d, "c", "Frozen primary contrast")
    panel_label(ax_c, "d", "Trajectory-level heterogeneity")
    panel_label(ax_e, "e", "Fixed utility sensitivity")
    save_figure(fig, OUTPUT / "figure_6_paired_formulation_boundary")
    plt.close(fig)


def build_architecture_boundary_figure() -> None:
    evidence_path = (
        ROOT
        / "experiment_results"
        / "v150_graph_vs_flat_confirmation_seed17400"
        / "V150_ARCHITECTURE_COMPARISON_AUDIT.json"
    )
    audit = json.loads(evidence_path.read_text(encoding="utf-8"))
    episode_rows = audit["episode_results"]
    paired = audit["paired_episode_bootstrap"]

    with (SOURCE_DATA / "figure_7_graph_vs_flat_by_trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    with (SOURCE_DATA / "figure_7_graph_vs_flat_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "mean_regret", "top1_agreement", "trainable_parameters"])
        writer.writerow(["Physics-graph", audit["graph_ranking"]["model_mean_regret"], audit["graph_ranking"]["top1_agreement"], 56457])
        writer.writerow(["Flat MLP", audit["flat_ranking"]["model_mean_regret"], audit["flat_ranking"]["top1_agreement"], 56457])
        writer.writerow(["DT-aware unchanged", audit["graph_ranking"]["baseline_mean_regret"], "", ""])
        writer.writerow(["Random choice", "", audit["graph_ranking"]["random_top1"], ""])
        writer.writerow(["Flat minus graph", paired["mean"], "", ""])
        writer.writerow(["95% CI low", paired["ci_low"], "", ""])
        writer.writerow(["95% CI high", paired["ci_high"], "", ""])

    names = ["Physics-graph", "Matched flat MLP", "DT-aware unchanged"]
    regrets = [
        audit["graph_ranking"]["model_mean_regret"],
        audit["flat_ranking"]["model_mean_regret"],
        audit["graph_ranking"]["baseline_mean_regret"],
    ]
    top1_names = ["Physics-graph", "Matched flat MLP", "Random choice"]
    top1 = [
        audit["graph_ranking"]["top1_agreement"],
        audit["flat_ranking"]["top1_agreement"],
        audit["graph_ranking"]["random_top1"],
    ]
    differences = np.array([row["flat_minus_graph_regret"] for row in episode_rows])
    episodes = np.arange(1, len(differences) + 1)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.75))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.95, 1.05), hspace=0.50, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    colors = [TEAL, ORANGE, GRAY]
    y = np.arange(3)[::-1]
    ax_a.barh(y, regrets, color=colors, height=0.62)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(names)
    ax_a.set_xlabel("Mean normalized ranking regret")
    ax_a.set_xlim(0, 0.49)
    ax_a.text(0.98, 0.96, "lower is better", transform=ax_a.transAxes, ha="right", va="top", color=GRAY, fontsize=5.8)
    for yi, value in zip(y, regrets):
        ax_a.text(value + 0.008, yi, f"{value:.3f}", va="center", fontsize=6.5, color=NAVY)
    ax_a.grid(axis="x", color=GRID, linewidth=0.6)

    ax_b.bar(np.arange(3), np.array(top1) * 100, color=colors, width=0.65)
    ax_b.set_xticks(np.arange(3))
    ax_b.set_xticklabels(["Physics-\ngraph", "Matched\nflat MLP", "Random\nchoice"])
    ax_b.set_ylabel("Top-1 agreement (%)")
    ax_b.set_ylim(0, 45)
    for x, value in enumerate(top1):
        ax_b.text(x, value * 100 + 1.1, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=6.5, color=NAVY)
    ax_b.text(0.98, 0.96, "higher is better", transform=ax_b.transAxes, ha="right", va="top", color=GRAY, fontsize=5.8)
    ax_b.grid(axis="y", color=GRID, linewidth=0.6)

    ax_c.bar(episodes, differences, color=np.where(differences > 0, TEAL, ORANGE), width=0.72)
    ax_c.axhline(0, color=NAVY, linewidth=0.8)
    ax_c.set_xticks(episodes)
    ax_c.set_xlabel("Independent confirmation trajectory")
    ax_c.set_ylabel("Flat minus graph regret")
    ax_c.set_ylim(-0.24, 0.32)
    ax_c.text(0.6, 0.295, "teal: graph lower", ha="left", va="top", color=TEAL, weight="bold", fontsize=5.2)
    ax_c.text(0.6, -0.222, "orange: flat lower", ha="left", va="bottom", color=ORANGE, weight="bold", fontsize=5.2)
    ax_c.text(12.4, 0.295, "7/12 favor graph", ha="right", va="top", color=NAVY, weight="bold", fontsize=5.5)
    ax_c.grid(axis="y", color=GRID, linewidth=0.6)

    mean = paired["mean"]
    low = paired["ci_low"]
    high = paired["ci_high"]
    ax_d.axvline(0, color=NAVY, linewidth=0.8)
    ax_d.errorbar(mean, 0.62, xerr=[[mean - low], [high - mean]], fmt="o", color=GRAY, ecolor=GRAY, capsize=4, markersize=6, elinewidth=1.5)
    ax_d.set_xlim(-0.075, 0.125)
    ax_d.set_ylim(0, 1)
    ax_d.set_yticks([])
    ax_d.set_xlabel("Paired flat-minus-graph regret difference")
    ax_d.grid(axis="x", color=GRID, linewidth=0.6)
    ax_d.text(0.5, 0.34, f"Mean {mean:+.3f}  ·  95% CI [{low:+.3f}, {high:+.3f}]", transform=ax_d.transAxes, ha="center", va="center", color=GRAY, weight="bold", fontsize=5.8)
    ax_d.text(
        0.5,
        0.10,
        "95% CI crosses zero",
        transform=ax_d.transAxes,
        ha="center",
        va="center",
        fontsize=5.4,
        color=RED,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F9E8E4", "edgecolor": "none"},
    )

    panel_label(ax_a, "a", "Mean ranking regret")
    panel_label(ax_b, "b", "Top-1 action agreement")
    panel_label(ax_c, "c", "Paired trajectory differences")
    panel_label(ax_d, "d", "Bootstrap estimate of flat-minus-graph regret")
    save_figure(fig, OUTPUT / "figure_7_graph_representation_boundary")
    plt.close(fig)


def build_anylogic_validation_figure() -> None:
    comparison_path = ROOT / "paper_outputs" / "anylogic_validation" / "final" / "python_anylogic_comparison.csv"
    with comparison_path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    (SOURCE_DATA / "figure_8_python_anylogic_comparison.csv").write_text(
        comparison_path.read_text(encoding="utf-8-sig"), encoding="utf-8"
    )

    by_scenario = {
        scenario: sorted((row for row in rows if row["scenario"] == scenario), key=lambda row: float(row["horizon_h"]))
        for scenario in ("steady", "rush")
    }
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.9))
    grid = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.34)
    axes = [fig.add_subplot(grid[index // 2, index % 2]) for index in range(4)]

    scenario_colors = {"steady": BLUE, "rush": RED}

    def plot_metric(axis, metric: str, ylabel: str) -> None:
        for scenario in ("steady", "rush"):
            scenario_rows = by_scenario[scenario]
            horizon = np.array([float(row["horizon_h"]) for row in scenario_rows])
            for platform, marker, linestyle, alpha in (
                ("anylogic", "o", "-", 1.0),
                ("python", "s", "--", 0.70),
            ):
                mean = np.array([float(row[f"{metric}_mean_{platform}"]) for row in scenario_rows])
                low = np.array([float(row[f"{metric}_ci95_low_{platform}"]) for row in scenario_rows])
                high = np.array([float(row[f"{metric}_ci95_high_{platform}"]) for row in scenario_rows])
                axis.errorbar(
                    horizon,
                    mean,
                    yerr=np.vstack([mean - low, high - mean]),
                    color=scenario_colors[scenario],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.2,
                    markersize=4,
                    capsize=2.5,
                    alpha=alpha,
                )
        axis.set_xticks([1, 4, 8])
        axis.set_xlabel("Physical horizon (h)")
        axis.set_ylabel(ylabel)
        axis.grid(color=GRID, linewidth=0.55)

    plot_metric(axes[0], "uph", "Throughput (UPH)")
    plot_metric(axes[1], "avg_waiting_time_min", "Mean waiting time (min)")
    plot_metric(axes[2], "unfinished_tasks", "Backlog at horizon")

    ax_d = axes[3]
    all_uph = []
    for scenario in ("steady", "rush"):
        scenario_rows = by_scenario[scenario]
        x = np.array([float(row["uph_mean_anylogic"]) for row in scenario_rows])
        y = np.array([float(row["uph_mean_python"]) for row in scenario_rows])
        horizon = [int(float(row["horizon_h"])) for row in scenario_rows]
        ax_d.scatter(x, y, s=34, color=scenario_colors[scenario], edgecolor="white", linewidth=0.7, label=scenario.capitalize(), zorder=3)
        label_offsets = {
            "steady": {1: (0.45, -0.75, "left"), 4: (0.45, 0.35, "left"), 8: (0.45, 0.35, "left")},
            "rush": {1: (0.45, 0.35, "left"), 4: (0.55, -1.25, "left"), 8: (-0.55, 0.60, "right")},
        }
        for xi, yi, hour in zip(x, y, horizon):
            dx, dy, alignment = label_offsets[scenario][hour]
            ax_d.text(xi + dx, yi + dy, f"{hour} h", fontsize=5.0, color=scenario_colors[scenario], ha=alignment)
        all_uph.extend(x.tolist())
        all_uph.extend(y.tolist())
    minimum, maximum = min(all_uph) - 2, max(all_uph) + 2
    ax_d.plot([minimum, maximum], [minimum, maximum], color=GRAY, linestyle="--", linewidth=0.9, zorder=1)
    ax_d.set_xlim(minimum, maximum)
    ax_d.set_ylim(minimum, maximum)
    ax_d.set_xlabel("AnyLogic throughput (UPH)")
    ax_d.set_ylabel("Python throughput (UPH)")
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.grid(color=GRID, linewidth=0.55)
    max_relative = max(abs(float(row["uph_relative_difference_pct"])) for row in rows)
    ax_d.text(
        0.04,
        0.06,
        f"max |relative difference| = {max_relative:.1f}%",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        color=TEAL,
        weight="bold",
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.4},
    )

    titles = (
        "Throughput across physical horizons",
        "Waiting-time growth under load",
        "End-of-horizon backlog",
        "Cross-platform throughput agreement",
    )
    for label, axis, title in zip("abcd", axes, titles):
        panel_label(axis, label, title)

    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="o", linestyle="-", label="AnyLogic steady"),
        Line2D([0], [0], color=BLUE, marker="s", linestyle="--", alpha=0.70, label="Python steady"),
        Line2D([0], [0], color=RED, marker="o", linestyle="-", label="AnyLogic rush"),
        Line2D([0], [0], color=RED, marker="s", linestyle="--", alpha=0.70, label="Python rush"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=5.8, columnspacing=1.2, handlelength=2.2)
    save_figure(fig, OUTPUT / "figure_8_anylogic_validation")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    build_framework_figure()
    build_cad_scene_figure()
    build_detailed_model_figure_jms()
    build_physics_factorial_figure()
    build_decision_evidence_figure()
    build_paired_formulation_figure()
    build_architecture_boundary_figure()
    build_anylogic_validation_figure()
    print(f"Submission figures saved to {OUTPUT}")


if __name__ == "__main__":
    main()
