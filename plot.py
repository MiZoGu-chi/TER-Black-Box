import matplotlib.pyplot as plt

def plot_learning_curve(mses):
    plt.figure(figsize=(12,5))

    plt.subplot(1, 1, 1)
    plt.plot(mses, color='red', linewidth=2)
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Itération")
    plt.ylabel("MEAN SQUARED ERROR")
    plt.grid(True)



def plot_learning_curve_mae(maes):
    plt.figure(figsize=(12,5))

    plt.subplot(1, 1, 1)
    plt.plot(maes, color='red', linewidth=2)
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Itération")
    plt.ylabel("MEAN ABSOLUTE ERROR")
    plt.grid(True)


def plot_compare_mae_lars_cg(maes_lars, maes_cg, D, instance_name):
    plt.figure(figsize=(10, 6))

    plt.plot(maes_cg, color='red', marker='.', linewidth=1.5, label='CG')
    plt.plot(maes_lars, color='blue', marker='.', linewidth=1.5, label='LARS')

    plt.title(f"LARS vs CG (D={D})\nInstance : {instance_name}", fontsize=16)
    plt.xlabel("Sample size", fontsize=12)
    plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
    
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

    plt.legend(title='method', loc='upper right', fontsize=12)

def plot_compare_models_mae(models, D, instance_name):
    plt.figure(figsize=(10, 6))

    for model_name, maes in models.items():
        plt.plot(maes, linewidth=2.0, alpha=0.8, label=model_name)

    plt.title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)
    plt.xlabel("Sample size", fontsize=12)
    plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
    
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

    plt.legend(title='method', loc='upper right', fontsize=12)

    #plt.yscale('log')

    plt.tight_layout()

    #plt.ylim(-0.05, 1.0)