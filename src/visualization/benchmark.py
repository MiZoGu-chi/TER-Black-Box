import matplotlib.pyplot as plt

from .utils import figure


def plot_compare_models(models, D, instance_name, loss_function):
    with figure(instance_name, f"D={D}", loss_function):
        plt.figure(figsize=(10, 6))
        for m, info in models.items():
            plt.plot(info['loss_values'], linewidth=2.0, alpha=0.8, label=info['name'], color=info['color'])
        plt.title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)
        plt.xlabel("Sample size", fontsize=12)
        plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
        plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
        plt.legend(title='method', loc='upper right', fontsize=12)
        plt.tight_layout()
