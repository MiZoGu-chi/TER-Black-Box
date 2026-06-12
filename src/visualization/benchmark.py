import matplotlib.pyplot as plt

from .utils import figure


def plot_compare_models(models, N, K, num, D, loss_function, seed, models_list):
    models_tag = "-".join(models_list)
    filename = f"{N}_{K}_{num}_{D}_{loss_function}_{seed}_{models_tag}"
    with figure(filename, subdir=f"benchmark/{N}/{K}", overwrite=True):
        plt.figure(figsize=(10, 6))
        for m, info in models.items():
            plt.plot(info['loss_values'], linewidth=2.0, alpha=0.8, label=info['name'], color=info['color'])
        plt.xlabel("Sample size", fontsize=12)
        plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
        plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
        plt.legend(title='method', loc='upper right', fontsize=12)
        plt.tight_layout()
