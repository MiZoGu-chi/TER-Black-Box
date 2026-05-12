import matplotlib.pyplot as plt

import os

from io_utils import OptimizationResult


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

    save_plot(instance_name, f"D={D}", loss_function)


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

# à relire
def plot_optimization(filepath):
    result = OptimizationResult(filepath)
    
    m = result.meta
    instance_name = f"opti_N{m['N']}_K{m['K']}_D{m['D']}_N{m['num']}_S{m['seed']}_{m['algo']}_{m['surrogate']}"
    
    best_fit = [item[0] for item in result.fitness_history]
    evals = [item[1] for item in result.fitness_history] 
    
    pred_fit = [item[0] for item in result.fidelity_history]
    real_fit = [item[1] for item in result.fidelity_history]   
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(evals, best_fit, marker='.', linestyle='-', color='b')
    axes[0].set_title("Convergence de l'optimisation")
    axes[0].set_xlabel("Évaluations") 
    axes[0].set_ylabel("Meilleure Fitness")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].scatter(real_fit, pred_fit, s=10, color='black', alpha=0.5)
    if real_fit and pred_fit:
        min_val = min(min(real_fit), min(pred_fit))
        max_val = max(max(real_fit), max(pred_fit))
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
        
    axes[1].set_title("Précision du modéle surrogate")
    axes[1].set_xlabel("fitness")
    axes[1].set_ylabel("fitness prédite")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(instance_name, "analyse_visuelle")
    plt.close()

def plot_aggregated_fitness_distributions(max_fitness_by_K):
    plt.figure(figsize=(8, 6))
    
    sorted_K = sorted(max_fitness_by_K.keys())
    fitness_distributions = [max_fitness_by_K[k] for k in sorted_K]
    
    plt.boxplot(fitness_distributions, labels=[f"K={k}" for k in sorted_K])
    
    plt.title("Distribution de la fitness par valeur de K", fontsize=14)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_plot("aggregated_fitness")
    plt.close()    

def plot_aggregated_convergence(fitness_trajectories_by_K):

    plt.figure(figsize=(10, 6))
    
    colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}
    
    for K, trajectories in fitness_trajectories_by_K.items():
        color = colors.get(K)
        
        # On trace les lignes sans label d'abord
        for run in trajectories:
            fitness = [item[0] for item in run]
            n_evals = [item[1] for item in run]
            plt.plot(n_evals, fitness, color=color, alpha=0.2)
            
    plt.title("Convergence de l'optimisation agrégé pour chaque valeur de K", fontsize=14)
    plt.xlabel("Nombre d'évalution", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_plot("optimization_convergence", "trajectories")
    plt.close()

def plot_aggregated_fidelity(fidelity_samples_by_K):
    plt.figure(figsize=(8, 8))
    
    colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}
    
    reals_fitness = []
    preds_fitness = []

    for K, samples in fidelity_samples_by_K.items():
        preds = [s[0] for s in samples]
        reals = [s[1] for s in samples]
        
        reals_fitness.extend(reals)
        preds_fitness.extend(preds)
        
        plt.scatter(reals, preds, s=5, color=colors.get(K), alpha=0.3, label=f"K={K}")

    if reals_fitness and preds_fitness:
        min_val = min(min(reals_fitness), min(preds_fitness))
        max_val = max(max(reals_fitness), max(preds_fitness))
        plt.plot([min_val, max_val], [min_val, max_val], 'b--')

    plt.title("Précision globale du modèle surrogate par K", fontsize=14)
    plt.xlabel("Fitness réelle", fontsize=12)
    plt.ylabel("Fitness prédite", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot("model_fidelity")
    plt.close()

def save_df_as_image(df):
    height = 0.5 * (len(df) + 1) 
    fig, ax = plt.subplots(figsize=(10, height)) 
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    save_plot("interaction metrics")
    plt.close()

def save_plot(base_name, *tags):
    filename = base_name

    for tag in tags:
        if tag:
            filename += f"_{tag}"
            
    extension = ".png"
    path = f"plots/{filename}{extension}"
    
    counter = 1
    while os.path.exists(path):
        path = f"plots/{filename}_{counter}{extension}"
        counter += 1
    
    plt.savefig(path)
    print(f"Graphique sauvegardé : {path}")