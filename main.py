import argparse
import warnings
import random
import numpy as np
from sklearnex import patch_sklearn
from sklearn.exceptions import ConvergenceWarning

from commands import OptimizeCommand, CompareCommand
from models import MODELS
from algorithm import ALGORITHMS, START_STRATEGIES

def setup_environment():
    patch_sklearn()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    setup_environment()

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument("N", type=int)
    parser_compare.add_argument("K", type=int)
    parser_compare.add_argument("--NB_ENTRAINEMENT", type=int, default=400)
    parser_compare.add_argument("--NB_TEST", type=int, default=1000)
    parser_compare.add_argument("--models", nargs='+', choices=MODELS.keys(), default=['cg', 'lasso_lars'])
    parser_compare.add_argument("--loss_function", type=str, default='mae', choices=['mae', 'mse'])

    parser_optimize = subparsers.add_parser('optimize')
    parser_optimize.add_argument("--budget", type=int, default=1000)
    parser_optimize.add_argument("--initial_pop_size", type=int, default=20)
    parser_optimize.add_argument("--surrogate", type=str, default='lasso_lars', choices=MODELS.keys())
    parser_optimize.add_argument("--algo", type=str, default='tabu', choices=ALGORITHMS.keys())
    parser_optimize.add_argument("--start", type=str, default='rd', choices=START_STRATEGIES.keys())

    args = parser.parse_args()

    if args.seed is not None: 
        np.random.seed(args.seed)
        random.seed(args.seed)

    commands = {
        'compare': CompareCommand,
        'optimize': OptimizeCommand
    }

    # Exécution
    if args.mode in commands:
        cmd = commands[args.mode](args)
        cmd.execute()