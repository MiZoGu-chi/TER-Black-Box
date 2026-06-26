import argparse
import warnings
import random
import numpy as np

from sklearnex import patch_sklearn
from sklearn.exceptions import ConvergenceWarning

from src.cli.commands import OptimizeCommand, CompareCommand, AnalyzeCommand, EvaluateCommand, RankCommand

def setup_environment():
    patch_sklearn()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":
    setup_environment()

    commands = {
        'compare': CompareCommand,
        'optimize': OptimizeCommand,
        'analyze': AnalyzeCommand,
        'evaluate': EvaluateCommand,
        'rank': RankCommand,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name, command_cls in commands.items():
        command_cls.add_arguments(subparsers.add_parser(name))

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    commands[args.mode](args).execute()
