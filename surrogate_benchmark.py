import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_sampler import generate_dataset
from plot import plot_compare_models

def run_model_comparison(args, landscape, masks, models_dict):
    print("Génération du dataset d'entraînement...")
    X_train, Y_train = generate_dataset(args.N, args.NB_ENTRAINEMENT, masks, landscape.eval)

    print("Génération du dataset de test...")
    X_test, Y_test = generate_dataset(args.N, args.NB_TEST, masks, landscape.eval)

    res = {model_name: [] for model_name in args.models}
    
    START_SIZE = 50 
    STEP_SIZE = 1

    for t in range(START_SIZE, args.NB_ENTRAINEMENT, STEP_SIZE):
        for model_name in args.models:
            surrogate = models_dict[model_name]['model']
            
            surrogate.fit(X_train[:(t+1)], Y_train[:(t+1)])
            predictions = surrogate.predict(X_test)
            
            match args.loss_function:
                case 'mae':
                    loss_value = mean_absolute_error(Y_test, predictions)
                case 'mse':
                    loss_value = mean_squared_error(Y_test, predictions)
                
            res[model_name].append(loss_value)
            print(f"{model_name} - Itération : {t} ; {args.loss_function.upper()} : {loss_value}")
    
    res_display = {
        m: {
            'name': models_dict[m]['name'], 
            'color': models_dict[m]['color'],
            'loss_values': res[m]
        } for m in args.models
    }

    plot_compare_models(res_display, args.D, f"nk_{args.N}_{args.K}_{args.num}", args.loss_function) 
    plt.show()