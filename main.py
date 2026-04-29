import argparse

import os

import numpy as np

import random

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
from algorithm import ALGORITHMS, START_STRATEGIES

import warnings
from sklearn.exceptions import ConvergenceWarning

from joblib import Parallel, delayed

from datetime import datetime

import glob # sert à utiliser *

from analyze import analyser_fichier

warnings.filterwarnings("ignore", category=ConvergenceWarning) # Sinon le terminal est pollué

def task_wrapper(N, K, seed, args, output_dir):
    np.random.seed(seed)
    random.seed(seed)

    args.N = N
    args.K = K

    instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{args.num}.txt"

    if not os.path.exists(instance_name):
        return

    landscape = NK_Landscape(instance_name)
    masks = generate_masks(N, args.D)
    model = MODELS[args.surrogate]['model']
    
    best_fit, best_sol, history, surrogate_model = run_surrogate_optimization(args, landscape, masks, model, ALGORITHMS[args.algo], START_STRATEGIES[args.start])

    coefs = surrogate_model.coef_.tolist()

    filename = f"{output_dir}/opti_{N}_{K}_{seed}_{args.algo}_{args.surrogate}.txt"

    with open(filename, "w") as f:
        f.write(f"{N} {K} {seed} {args.algo} {args.surrogate}\n")
        
        for sol, budg in history:
            f.write(f"{sol} {budg}\n")
            
        f.write(f"{coefs}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument("--NB_ENTRAINEMENT", type=int, default=400)
    parser_compare.add_argument("--NB_TEST", type=int, default=1000)
    parser_compare.add_argument("--models", nargs='+', choices=MODELS.keys(), default=['cg', 'lasso_lars'])
    parser_compare.add_argument("--loss_function", type=str, default='mae', choices=['mae', 'mse'])

    parser_optimize = subparsers.add_parser('optimize')
    parser_optimize.add_argument("--budget", type=int, default=100)
    parser_optimize.add_argument("--initial_pop_size", type=int, default=20)
    #parser_optimize.add_argument("--window_size", type=int, default=30)
    parser_optimize.add_argument("--surrogate", type=str, default='lasso_lars_ic', choices=MODELS.keys())
    parser_optimize.add_argument("--algo", type=str, default='tabu', choices=ALGORITHMS.keys())
    parser_optimize.add_argument("--start", type=str, default='rd', choices=START_STRATEGIES.keys())
    args = parser.parse_args()


    instance_name = f"instances/{args.N}/{args.K}/nk_{args.N}_{args.K}_{args.num}.txt"

    if not os.path.exists(instance_name):
        print(f"instance {instance_name} inexistante...")
        exit()

    if args.seed is not None: 
        np.random.seed(args.seed) # np.random.randint
        random.seed(args.seed) # random.random()

    landscape = NK_Landscape(instance_name)
    print(f"Instance chargée : {instance_name}")
    

    print("Génération des masques...")
    masks = generate_masks(args.N, args.D)

    match args.mode:
        case 'compare':
            run_model_comparison(args, landscape, masks, MODELS)
            
        case 'optimize':
            #model = MODELS[args.surrogate]['model']
            #run_surrogate_optimization(args, landscape, masks, model, ALGORITHMS[args.algo], START_STRATEGIES[args.start])
            configs_N = [64, 128, 256]
            configs_K = [1, 2, 4, 8]
            config_seeds = range(10)
            n_jobs_=-1

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"opti_output/run_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            Parallel(n_jobs=n_jobs_)(delayed(task_wrapper)(N, K, seeds, args, output_dir) for N in configs_N for K in configs_K for seeds in config_seeds)

            fichiers_resultats = glob.glob(f"{output_dir}/opti_*.txt")

            for fichier in fichiers_resultats:
                stats = analyser_fichier(fichier)
                
                if stats is not None:
                    pourcentage, nb_zeros, nb_total = stats
                    nom_court = os.path.basename(fichier)
                    print(f"[{nom_court}] : {pourcentage:.1f}% de zéros ({nb_zeros}/{nb_total})")
            
        case _:
            print(f"Mode '{args.mode}' non reconnu.")
