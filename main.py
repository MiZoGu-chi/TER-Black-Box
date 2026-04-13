import argparse

import os

import numpy as np

from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsIC

from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from data_sampler import generate_dataset
from walsh import generate_masks, apply_walsh
from nk import NK_Landscape  

from plot import plot_compare_models


MODELS = {
    #'Lars' : Lars(fit_intercept=False),
    'LARS (Lasso)': LassoLars(alpha=0.001, fit_intercept=False),
    'LARS (LassoIc)': LassoLarsIC(criterion='bic', fit_intercept=False, noise_variance=1e-5),
    'CG (RIDGE)': Ridge(alpha=0.001, fit_intercept=False, solver='sparse_cg'),
    'ElasticNet': ElasticNet(alpha=0.001, fit_intercept=False),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--NB_ENTRAINEMENT", type=int, default=400)
    parser.add_argument("--NB_TEST", type=int, default=1000)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("num", type=int)

    parser.add_argument("--D", type=int, default=2)
    
    parser.add_argument("--models", nargs='+', choices=MODELS.keys(), default=list(MODELS.keys()))
    parser.add_argument("--loss_function", type=str, default='mae', choices=['mae', 'mse'])
    
    args = parser.parse_args()

    instance_name = f"instances/{args.N}/{args.K}/nk_{args.N}_{args.K}_{args.num}.txt"

    if not os.path.exists(instance_name):
        print(f"instance {instance_name} inexistante...")
        exit()

    print(f"instance chargé'e : {instance_name}")
    
    landscape = NK_Landscape(instance_name)

    print("generation des masques...")
    masks = generate_masks(args.N, args.D)

    print("generation du dataset d'entraînement...")
    X_train, Y_train = generate_dataset(args.N, args.NB_ENTRAINEMENT, masks, landscape.eval)

    print("génération du datast de test...")
    X_test, Y_test = generate_dataset(args.N, args.NB_TEST, masks, landscape.eval)

    res = {model_name: [] for model_name in args.models}
  
    for t in range(args.NB_ENTRAINEMENT):
        
        for model_name in args.models:
            surrogate = MODELS[model_name]
            
            surrogate.fit(X_train[:(t+1)], Y_train[:(t+1)])
            
            predictions = surrogate.predict(X_test)
            
            match args.loss_function:
                case 'mae':
                    loss_value = mean_absolute_error(Y_test, predictions)
                case 'mse':
                    loss_value = mean_squared_error(Y_test, predictions)
                
            res[model_name].append(loss_value)

            print(f"{model_name} - Itération : {t} ; {args.loss_function.upper()} : {loss_value}")

    #model = args.models[0]
    #surrogate = MODELS[model]
    #preds = surrogate.predict(X_test)
    #plot_compare_models(res, args.D, f"nk_{args.N}_{args.K}_{args.num}", Y_test, preds) 
    
    plot_compare_models(res, args.D, f"nk_{args.N}_{args.K}_{args.num}") 

    if args.loss_function == 'mse':
        plt.ylabel("Mean Squared Error of fitness", fontsize=12)
        
    plt.show()