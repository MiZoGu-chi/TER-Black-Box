import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_compare_and_scatter_models(models, D, instance_name, f, f_hat, loss_function="mae"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, maes in models.items():
        axes[0].plot(maes, linewidth=2.0, alpha=0.8, label=model_name)

    axes[0].set_title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)
    axes[0].set_xlabel("Sample size", fontsize=12)
    axes[0].set_ylabel("Mean Abs. Error of fitness", fontsize=12)
    
    axes[0].grid(True, color='lightgray', linestyle='-', linewidth=0.5)

    axes[0].legend(title='method', loc='upper right', fontsize=12)

    axes[0].set_yscale('log')

    axes[0].set_ylim(-0.05, 1.0)

    axes[1].scatter(f, f_hat, s=2, color='black', alpha=0.5)

    axes[1].set_xlabel("f(x)")
    axes[1].set_ylabel("Approximation of f(x)")
    
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    save_plot(instance_name, D, loss_function)

 
def plot_compare_models(models, D, instance_name, loss_function):
    plt.figure(figsize=(10, 6))

    for m, info in models.items():
        plt.plot(info['loss_values'], linewidth=2.0, alpha=0.8, label=info['name'], color=info['color'])

    plt.title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)

    plt.xlabel("Sample size", fontsize=12)
    plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    plt.legend(title='method', loc='upper right', fontsize=12)
    
    plt.tight_layout()

    save_plot(instance_name, D, loss_function) 

    # pour avoir en parrallèle l'échelle log

    # plt.figure(figsize=(10, 6))

    # for model_name, maes in models.items():
    #     plt.plot(maes, linewidth=2.0, alpha=0.8, label=model_name)

    # plt.title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)

    # plt.xlabel("Sample size", fontsize=12)
    # plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
    # plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    # plt.legend(title='method', loc='upper right', fontsize=12)

    # plt.yscale('log')
    # plt.tight_layout()

    # save_plot(instance_name, D, loss_function + "_log")

def plot_convergence_comparison(results, baseline_score=None):
    plt.figure(figsize=(10, 6))
    
    # Si on reçoit une simple liste (un seul résultat)
    if isinstance(results, list):
        plt.plot(results, label='Optimization Run', linewidth=2)
    elif isinstance(results, dict):
        for label, history in results.items():
            plt.plot(history, label=label, linewidth=2)
    
    if baseline_score is not None:
        plt.axhline(y=baseline_score, color='red', linestyle='--', 
                    label=f'Baseline: {baseline_score:.4f}')
    
    plt.title('Assisted surrogate convergence')
    plt.xlabel('Budget')
    plt.ylabel('Best Fitness')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def save_plot(instance_name, D, loss_function):
    filename = f"{instance_name}_D={D}_{loss_function}"
    extension = ".png"
    path = f"plots/{filename}{extension}"
    
    counter = 1
    while os.path.exists(path):
        path = f"plots/{filename}_{counter}{extension}"
        counter += 1
    
    plt.savefig(path)
    print(f"graphique sauvegardé : {path}")