import argparse
import warnings
import random
import numpy as np

from sklearnex import patch_sklearn
from sklearn.exceptions import ConvergenceWarning

from src.cli.commands import OptimizeCommand, CompareCommand, AnalyzeCommand, EvaluateCommand
from src.core.models import MODELS
from src.core.algorithm import ALGORITHMS, START_STRATEGIES

def setup_environment():
    patch_sklearn()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    setup_environment()

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num", type=int, default=0) # à enlever
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
    parser_optimize.add_argument("--N", type=int, nargs='+', default=[64], choices=[64, 128, 256])
    parser_optimize.add_argument("--K", type=int, nargs='+', default=[1, 2], choices=[1, 2, 4, 8])
    parser_optimize.add_argument("--budget", type=int, default=1000)
    parser_optimize.add_argument("--initial_pop_size", type=int, default=20)
    parser_optimize.add_argument("--surrogate", type=str, default='lasso_lars', choices=MODELS.keys())
    parser_optimize.add_argument("--algo", type=str, default='tabu', choices=ALGORITHMS.keys())
    parser_optimize.add_argument("--start", type=str, default='rd', choices=START_STRATEGIES.keys())
    parser_optimize.add_argument("--alpha", type=float, default=0.001, help="alpha de lasso lars : terme de régularisation L1")
    parser_optimize.add_argument("--tabu_factor", type=int, default=10)
    instance_selector = parser_optimize.add_mutually_exclusive_group()
    instance_selector.add_argument("--instances", type=int, nargs='+', choices=range(10))
    instance_selector.add_argument("--n_instances", type=int, default=10, choices=range(1, 11))
    seed_selector = parser_optimize.add_mutually_exclusive_group()
    seed_selector.add_argument("--seeds", type=int, nargs='+')
    seed_selector.add_argument("--n_seeds", type=int, default=10)

    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.add_argument("path", type=str, nargs='+')

    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument("path", type=str)
    parser_evaluate.add_argument("--samples", type=int, default=1000)
    parser_evaluate.add_argument("--sampler", type=str, default='stratified', choices=['random', 'stratified', 'whitebox'])

    args = parser.parse_args()

    if args.seed is not None: 
        np.random.seed(args.seed)
        random.seed(args.seed)

    commands = {
        'compare': CompareCommand,
        'optimize': OptimizeCommand,
        'analyze': AnalyzeCommand,
        'evaluate' : EvaluateCommand
    }

    if args.mode in commands:
        cmd = commands[args.mode](args)
        cmd.execute()