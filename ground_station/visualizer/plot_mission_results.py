import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Get paths relative to this script's directory for robust execution
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "json_files")

# 1. Discover all matching JSON files
search_pattern = os.path.join(script_dir, "*.json")
good_files = glob.glob(search_pattern)

if not good_files:
    raise FileNotFoundError(f"Error: No JSON files found in: {script_dir}")

good_files.sort()
print(f"Found {len(good_files)} mission(s) for analysis.\n")

# Define the global output directory
output_dir = os.path.join(current_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# ── Global Plot Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":          8.0,
    "axes.labelsize":    8.0,
    "axes.titlesize":    8.0,
    "xtick.labelsize":   7.0,
    "ytick.labelsize":   7.0,
    "legend.fontsize":   7.0,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "axes.grid":         True,
    "grid.linewidth":    0.3,
    "grid.alpha":        0.4,
    "grid.color":        "#aaaaaa",
    "figure.dpi":        300,
})

# Harmonized, publication-ready color palette
BLUE   = "#2b5c8f"  # Slate Blue
RED    = "#c0392b"  # Deep Red (accent)
GREEN  = "#2e7d32"  # Forest Green
ORANGE = "#d84315"  # Burnt Orange
GRAY   = "#37474f"  # Dark Slate Gray
LIGHT  = "#bbdefb"  # Soft Light Blue
LGREEN = "#c8e6c9"  # Soft Light Green

# 2. Main execution loop over all discovered files
for json_path in good_files:
    json_basename = os.path.basename(json_path)
    mission_name, _ = os.path.splitext(json_basename)
    
    print(f"Processing: {json_basename} ...")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        run_data = data["runs"][0]
        feedback = run_data["feedback"]

        GT_X = run_data["ground_truth"]["x"]
        GT_Y = run_data["ground_truth"]["y"]

        xs    = [f["position"]["x"] for f in feedback]
        ys    = [f["position"]["y"] for f in feedback]
        iters = [f["iteration"] for f in feedback]
        dts   = [f["distance_travelled"] for f in feedback]
        covs  = [f["coverage_area_m2"] for f in feedback]
        maxP  = [f["max_posterior"] for f in feedback]
        ig_g  = [f["ig_geometric"] for f in feedback]
        ig_s  = [f["ig_semantic"] for f in feedback]
        vlmt  = [f["vlm_inference_time_sec"] for f in feedback]

        target_step = next((f["iteration"] for f in feedback if f.get("target_estimate") is not None and f["target_estimate"].get("x") is not None and f["target_estimate"].get("y") is not None), None)
        target_est  = next((f["target_estimate"] for f in feedback if f.get("target_estimate") is not None and f["target_estimate"].get("x") is not None and f["target_estimate"].get("y") is not None), None)


        total_dist  = run_data["result"]["distance_travelled"]
        total_time  = run_data["mission"]["finished_at"] - run_data["mission"]["started_at"]
        n_steps     = len(feedback)
        instruction = run_data["mission"]["instruction"]

        # ── Layout Configuration ──────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 5.8))
        gs  = GridSpec(3, 3, figure=fig,
                       left=0.08, right=0.97,
                       top=0.95,  bottom=0.15,
                       wspace=0.38, hspace=0.48)

        ax_traj = fig.add_subplot(gs[:2, 0])   # Left, mid-tall trajectory
        ax_post = fig.add_subplot(gs[0, 1])    # Top Center
        ax_cov  = fig.add_subplot(gs[0, 2])    # Top Right
        ax_ig   = fig.add_subplot(gs[1, 1])    # Mid Center
        ax_vlm  = fig.add_subplot(gs[1, 2])    # Mid Right
        ax_dist = fig.add_subplot(gs[2, :])    # Bottom wide column spanning the baseline!

        # ── (a) Trajectory Map ────────────────────────────────────────────────
        cmap   = plt.cm.Blues
        norm   = plt.Normalize(vmin=0, vmax=n_steps + 1)

        ax_traj.plot(xs, ys, color=GRAY, linewidth=0.8, linestyle="--", zorder=1, alpha=0.5)

        for i, (x, y) in enumerate(zip(xs, ys)):
            c = cmap(norm(i + 1))
            ax_traj.scatter(x, y, s=45, color=c, edgecolors=BLUE, linewidths=0.6, zorder=3)
            ax_traj.annotate(f"$w_{{{i+1}}}$", xy=(x, y), xytext=(4, 4),
                             textcoords="offset points", fontsize=6.0, color=GRAY)

        for i in range(len(xs) - 1):
            dx, dy = xs[i+1] - xs[i], ys[i+1] - ys[i]
            ax_traj.annotate("",
                xy=(xs[i] + 0.65*dx, ys[i] + 0.65*dy),
                xytext=(xs[i] + 0.35*dx, ys[i] + 0.35*dy),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.6, mutation_scale=6), zorder=2)

        ax_traj.scatter(xs[0], ys[0], s=65, marker="s", color=BLUE, zorder=4)
        ax_traj.scatter(xs[-1], ys[-1], s=75, marker="*", color=RED, zorder=4)

        if target_est:
            ax_traj.scatter(target_est["x"], target_est["y"], s=75, marker="^", color=ORANGE, zorder=5)
            ax_traj.scatter(GT_X, GT_Y, s=75, marker="P", color=GREEN, zorder=5)
            
            ax_traj.plot([target_est["x"], GT_X], [target_est["y"], GT_Y],
                         color=GRAY, linewidth=0.7, linestyle=":", zorder=4)
            err = np.hypot(target_est["x"] - GT_X, target_est["y"] - GT_Y)
            mid_x = (target_est["x"] + GT_X) / 2
            mid_y = (target_est["y"] + GT_Y) / 2
            ax_traj.annotate(r"$\epsilon = {:.2f}$ m".format(err), xy=(mid_x, mid_y),
                             xytext=(5, -8), textcoords="offset points",
                             fontsize=6.0, color=GRAY, weight='bold')

        # Compute dynamic bounds to keep the trajectory centered with equal aspect ratio
        traj_xs = xs + [GT_X]
        traj_ys = ys + [GT_Y]
        if target_est:
            traj_xs.append(target_est["x"])
            traj_ys.append(target_est["y"])

        min_x, max_x = min(traj_xs), max(traj_xs)
        min_y, max_y = min(traj_ys), max(traj_ys)
        
        dx = max(0.6, (max_x - min_x) * 0.15)
        dy = max(0.6, (max_y - min_y) * 0.15)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        max_range = max(max_x - min_x + 2*dx, max_y - min_y + 2*dy)

        ax_traj.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
        ax_traj.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.set_xlabel("$x$ [m]")
        ax_traj.set_ylabel("$y$ [m]")
        ax_traj.set_title("(a) Exploration trajectory", loc="left", fontweight="bold", pad=4)
        ax_traj.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax_traj.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # ── (b) Max Posterior ─────────────────────────────────────────────────
        # Highlight target detection step if valid, otherwise use standard color
        colors_post = [RED if (target_step is not None and it == target_step) else LIGHT for it in iters]
        bars = ax_post.bar(iters, maxP, color=colors_post, edgecolor=BLUE, linewidth=0.5, zorder=3)
        
        for bar, val in zip(bars, maxP):
            if val > 0.05: 
                ax_post.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                             f"{val:.2f}", ha="center", va="bottom", fontsize=6.0, color=GRAY)
                             
        ax_post.axhline(0.5, color=GRAY, linewidth=0.6, linestyle=":", alpha=0.6)
        ax_post.set_xlabel("Step $k$")
        ax_post.set_ylabel(r"$\max F_k$")
        ax_post.set_title("(b) Max posterior", loc="left", fontweight="bold", pad=4)
        ax_post.set_xticks(iters)
        ax_post.set_ylim(0, 1.15)

        # ── (c) Cumulative Coverage area ──────────────────────────────────────
        ax_cov.plot(iters, covs, color=GREEN, linewidth=1.2,
                    marker="o", markersize=3, markerfacecolor="white",
                    markeredgewidth=0.8, zorder=3)
        ax_cov.fill_between(iters, covs, alpha=0.1, color=GREEN)
        
        for i, (it, cv) in enumerate(zip(iters, covs)):
            if i % 2 == 0 or i == len(iters)-1: 
                ax_cov.annotate(f"{cv:.1f}", xy=(it, cv), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontsize=6.0, color=GRAY)
        ax_cov.set_xlabel("Step $k$")
        ax_cov.set_ylabel(r"Coverage [m$^2$]")
        ax_cov.set_title("(c) Cumulative coverage", loc="left", fontweight="bold", pad=4)
        ax_cov.set_xticks(iters)
        ax_cov.set_ylim(0, max(covs) * 1.15)

        # ── (d) Information Gain Breakdown ────────────────────────────────────
        w = 0.35
        x_pos = np.array(iters)
        ax_ig.bar(x_pos - w/2, ig_g, width=w, color=LIGHT, edgecolor=BLUE, linewidth=0.5, label="Geometric IG", zorder=3)
        ax_ig.bar(x_pos + w/2, ig_s, width=w, color=LGREEN, edgecolor=GREEN, linewidth=0.5, label="Semantic IG", zorder=3)
        ax_ig.set_xlabel("Step $k$")
        ax_ig.set_ylabel("Information Gain")
        ax_ig.set_title("(d) Information gain", loc="left", fontweight="bold", pad=4)
        ax_ig.set_xticks(iters)

        # ── (e) VLM Inference Time ────────────────────────────────────────────
        bars_vlm = ax_vlm.bar(iters, vlmt, color=LIGHT, edgecolor=BLUE, linewidth=0.5, zorder=3)
        
        for bar, val in zip(bars_vlm, vlmt):
            ax_vlm.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.1f}s", ha="center", va="bottom", fontsize=5.8, color=GRAY)
        ax_vlm.set_xlabel("Step $k$")
        ax_vlm.set_ylabel("Inference Time [s]")
        ax_vlm.set_title("(e) VLM inference time", loc="left", fontweight="bold", pad=4)
        ax_vlm.set_xticks(iters)
        ax_vlm.set_ylim(0, max(vlmt) * 1.2)

        # ── (f) Cumulative Distance Profile (Restored!) ───────────────────────
        ax_dist.plot(iters, dts, color=BLUE, linewidth=1.2,
                     marker="o", markersize=3, markerfacecolor="white",
                     markeredgewidth=0.8, zorder=3)
        ax_dist.fill_between(iters, dts, alpha=0.08, color=BLUE)
        ax_dist.axhline(total_dist, color=RED, linewidth=0.7, linestyle=":")
        ax_dist.set_xlabel("Step $k$")
        ax_dist.set_ylabel("Distance [m]")
        ax_dist.set_title("(f) Distance profile", loc="left", fontweight="bold", pad=3)
        ax_dist.set_xticks(iters)
        ax_dist.set_ylim(0, total_dist * 1.15)
        ax_dist.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # ── Clean Spines and Tick Layout for Academic Publications ────────────
        for ax in [ax_post, ax_cov, ax_ig, ax_vlm, ax_dist]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ── Global Unified Horizontal Bottom Legend ───────────────────────────
        legend_elements = [
            Line2D([0], [0], marker='s', color='none', markerfacecolor=BLUE, markersize=5, label='Start'),
            Line2D([0], [0], marker='*', color='none', markerfacecolor=RED, markersize=8, label='Target'),
            Line2D([0], [0], marker='^', color='none', markerfacecolor=ORANGE, markersize=6, label=r'Est. ($\hat{p}$)'),
            Line2D([0], [0], marker='P', color='none', markerfacecolor=GREEN, markersize=6, label=r'GT ($p^*$)'),
            mpatches.Patch(facecolor=LIGHT, edgecolor=BLUE, linewidth=0.5, label='Geometric IG'),
            mpatches.Patch(facecolor=LGREEN, edgecolor=GREEN, linewidth=0.5, label='Semantic IG'),
            Line2D([0], [0], color=GREEN, linewidth=1.2, marker='o', markersize=3, label='Coverage Profile'),
            Line2D([0], [0], color=BLUE, linewidth=1.2, marker='o', markersize=3, label='Distance Profile')
        ]

        # Sits in 1 neat line at the very bottom base of the generated PDF page
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            framealpha=0.9,
            edgecolor="#cccccc",
            handletextpad=0.4,
            columnspacing=2.0,
            fontsize=7.0
        )

        pdf_output_path = os.path.join(output_dir, f"{mission_name}.pdf")
        fig.savefig(pdf_output_path, bbox_inches="tight", dpi=300)
        
        plt.close(fig)
        print(f" -> Saved to: {pdf_output_path}")

    except json.JSONDecodeError:
        print(f" [Warning] Skipping '{json_basename}': Invalid JSON structural encoding.\n")
    except Exception as e:
        print(f" [Warning] Unexpected error processing '{json_basename}': {e}\n")

print("All tasks completed.")