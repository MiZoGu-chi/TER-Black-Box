import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.core.data_sampler import generate_dataset
from src.visualization import plot_compare_models

def run_model_comparison(landscape, masks, models_dict, model_names, n_train, n_test, loss_function, D, instance, seed):

    print("Génération du dataset d'entraînement...")
    X_train, Y_train = generate_dataset(landscape.N, n_train, masks, landscape.eval)

    print("Génération du dataset de test...")
    X_test, Y_test = generate_dataset(landscape.N, n_test, masks, landscape.eval)

    res = {model_name: [] for model_name in model_names}

    START_SIZE = 0
    STEP_SIZE = 1

    for t in range(START_SIZE, n_train, STEP_SIZE):
        for model_name in model_names:
            surrogate = models_dict[model_name]['model']

            surrogate.fit(X_train[:(t+1)], Y_train[:(t+1)])
            predictions = surrogate.predict(X_test)

            match loss_function:
                case 'mae':
                    loss_value = mean_absolute_error(Y_test, predictions)
                case 'mse':
                    loss_value = mean_squared_error(Y_test, predictions)

            res[model_name].append(loss_value)
            print(f"{model_name} - Itération : {t} ; {loss_function.upper()} : {loss_value}")

    res_display = {
        m: {
            'name': models_dict[m]['name'],
            'color': models_dict[m]['color'],
            'loss_values': res[m]
        } for m in model_names
    }

    plot_compare_models(res_display, landscape.N, landscape.K, instance, D, loss_function, seed, model_names)
    plt.show()
