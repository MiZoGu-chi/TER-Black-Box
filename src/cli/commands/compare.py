import os 
from .base import Command
from src.core.surrogate_benchmark import run_model_comparison
from src.core.walsh import generate_masks
from src.core.nk import NK_Landscape
from src.core.models import MODELS

class CompareCommand(Command):

    def __init__(self, args):
        super().__init__(args)
        self.N = args.N
        self.K = args.K
        self.num = args.num
        self.D = args.D

    def execute(self):

        instance_name = f"instances/{self.N}/{self.K}/nk_{self.N}_{self.K}_{self.num}.txt"

        if not os.path.exists(instance_name):
            print(f"L'instance {instance_name} est inexistante.")
            return

        landscape = NK_Landscape(instance_name)
        print(f"Instance chargée : {instance_name}")

        print(f"Génération des masques (D={self.D})...")
        masks = generate_masks(self.N, self.D)

        run_model_comparison(self.args, landscape, masks, MODELS)

        