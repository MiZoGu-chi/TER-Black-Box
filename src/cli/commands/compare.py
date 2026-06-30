import os 
from .base import Command
from src.core.surrogate_benchmark import run_model_comparison
from src.core.walsh import generate_masks
from src.core.nk import NK_Landscape
from src.utils.io_utils import instance_path
from src.core.models import MODELS

class CompareCommand(Command):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("N", type=int, help="paramètre N du NK-landscape")
        parser.add_argument("K", type=int, help="paramètre K du NK-landscape")
        parser.add_argument("--instance", type=int, default=0, help="Indice de l'instance de NK-landscape à charger (0-9)")
        parser.add_argument("--D", type=int, default=2, help="Ordre de troncature du modèle de substitution de Walsh")
        parser.add_argument("--n_train", type=int, default=400, help="Nombre de solutions générées pour entraîner les modèles de régression")
        parser.add_argument("--n_test", type=int, default=1000, help="Nombre de solutions générées pour évaluer les modèles de régression")
        parser.add_argument("--models", nargs='+', choices=MODELS.keys(), default=['lasso_lars'], help="Modèles de régression à comparer")
        parser.add_argument("--loss_function", type=str, default='mae', choices=['mae', 'mse'], help="Fonction de perte utilisée pour comparer les modèles")

    def __init__(self, args):
        super().__init__(args)
        self.N = args.N
        self.K = args.K
        self.instance = args.instance
        self.D = args.D
        self.n_train = args.n_train
        self.n_test = args.n_test
        self.models = args.models
        self.loss_function = args.loss_function
        self.seed = args.seed

    def execute(self):

        instance_name = instance_path(self.N, self.K, self.instance)

        if not os.path.exists(instance_name):
            print(f"L'instance {instance_name} est inexistante.")
            return

        landscape = NK_Landscape(instance_name)
        print(f"Instance chargée : {instance_name}")

        print(f"Génération des masques (D={self.D})...")
        masks = generate_masks(self.N, self.D)
        print(f"Masques générés...")

        run_model_comparison(
            landscape, 
            masks, 
            MODELS, 
            self.models, 
            self.n_train, 
            self.n_test,
            self.loss_function, 
            self.D, 
            self.instance, 
            self.seed,
        )

        