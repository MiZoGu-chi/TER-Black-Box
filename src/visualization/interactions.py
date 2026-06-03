import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .utils import figure


def plot_confusion_metrics_distribution(all_matrices_by_K, N, surrogate=None, algo=None, subdir=None):
    from src.core.evaluation import compute_confusion_metrics

    sorted_K = sorted(all_matrices_by_K.keys())
    if not sorted_K:
        return

    with figure("confusion_distribution", subdir=subdir):
        fig, axes = plt.subplots(1, len(sorted_K), figsize=(6 * len(sorted_K), 5), squeeze=False)

        for ax, K in zip(axes[0], sorted_K):
            matrices = all_matrices_by_K[K]
            n_runs = len(matrices)
            D = matrices[0].shape[0]

            all_tp = np.zeros((n_runs, D))
            all_fp = np.zeros((n_runs, D))
            all_fn = np.zeros((n_runs, D))

            for i, m in enumerate(matrices):
                tp, fp, fn, _, _, _ = compute_confusion_metrics(m)
                all_tp[i] = tp
                all_fp[i] = fp
                all_fn[i] = fn

            positions, data, colors = [], [], []
            x_tick_pos, x_tick_labels = [], []

            for d in range(D):
                base = d * 3
                positions += [base, base + 1]
                data += [all_tp[:, d], all_fp[:, d]]
                colors += ['#2ecc71', '#e74c3c']
                x_tick_pos.append(base + 0.5)
                x_tick_labels.append(f"Ordre {d + 1}")

            bp = ax.boxplot(data, positions=positions, widths=0.6,
                            patch_artist=True, showfliers=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            for d in range(D):
                total_real = int(np.median(all_tp[:, d] + all_fn[:, d]))
                ax.plot(d * 3, total_real, marker='*', color='red',
                        markersize=13, zorder=5, linestyle='None')

            ax.set_xticks(x_tick_pos)
            ax.set_xticklabels(x_tick_labels)
            ax.set_ylabel("Nombre d'interactions", fontsize=11)
            info = "  ".join(filter(None, [surrogate, algo]))
            ax.set_title(f"K={K}  N={N}  {info}  {n_runs} runs", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            legend_elements = [
                Patch(facecolor='#2ecc71', alpha=0.6, label='TP'),
                Patch(facecolor='#e74c3c', alpha=0.6, label='FP'),
                Line2D([0], [0], marker='*', color='red', markersize=10,
                       linestyle='None', label='Total réel'),
            ]
            ax.legend(handles=legend_elements, fontsize=9)

        fig.suptitle(
            "Comparaison des interactions détectées par le surrogate\n"
            "avec les interactions réelles du paysage NK",
            fontsize=13
        )
        plt.tight_layout()
