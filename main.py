import argparse

import os

import numpy as np


from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import Lars, LassoLars, LassoLarsIC, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

from data_sampler import generate_dataset
from walsh import generate_masks, apply_walsh, apply_walsh_vectorized
from nk import NK_Landscape  

from plot import plot_compare_models

from surrogate_benchmark import run_model_comparison
from optimization import run_surrogate_optimization

from models import MODELS
from algorithm import ALGORITHMS

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning) # Sinon le terminal est pollué

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("num", type=int)
    parser.add_argument("--D", type=int, default=2)
    
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument("--NB_ENTRAINEMENT", type=int, default=400)
    parser_compare.add_argument("--NB_TEST", type=int, default=1000)
    parser_compare.add_argument("--models", nargs='+', choices=MODELS.keys(), default=['cg', 'lasso_lars'])
    parser_compare.add_argument("--loss_function", type=str, default='mae', choices=['mae', 'mse'])

    parser_optimize = subparsers.add_parser('optimize')
    parser_optimize.add_argument("--budget", type=int, default=1000)
    parser_optimize.add_argument("--initial_pop_size", type=int, default=20)
    #parser_optimize.add_argument("--window_size", type=int, default=30)
    parser_optimize.add_argument("--surrogate", type=str, default='lasso_lars_ic', choices=MODELS.keys())
    parser_optimize.add_argument("--algo", type=str, default='tabu', choices=ALGORITHMS.keys())
    args = parser.parse_args()

    instance_name = f"instances/{args.N}/{args.K}/nk_{args.N}_{args.K}_{args.num}.txt"

    if not os.path.exists(instance_name):
        print(f"instance {instance_name} inexistante...")
        exit()

    landscape = NK_Landscape(instance_name)
    print(f"Instance chargée : {instance_name}")
    

    print("Génération des masques...")
    masks = generate_masks(args.N, args.D)

    match args.mode:
        case 'compare':
            run_model_comparison(args, landscape, masks, MODELS)
            
        case 'optimize':
            model = MODELS[args.surrogate]['model']
            run_surrogate_optimization(args, landscape, masks, model, ALGORITHMS[args.algo])
            
        case _:
            print(f"Mode '{args.mode}' non reconnu.")
