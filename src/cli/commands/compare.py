import os 
from .base import Command
from src.core.surrogate_benchmark import run_model_comparison
from src.core.walsh import generate_masks
from src.core.nk import NK_Landscape
from src.core.models import MODELS

class CompareCommand(Command):
    def execute(self):

        instance_name = f"instances/{self.args.N}/{self.args.K}/nk_{self.args.N}_{self.args.K}_{self.args.num}.txt"

        if not os.path.exists(instance_name):
            print(f"L'instance {instance_name} est inexistante.")
            return  

        landscape = NK_Landscape(instance_name)
        print(f"Instance chargée : {instance_name}")

        print(f"Génération des masques (D={self.args.D})...")
        masks = generate_masks(self.args.N, self.args.D)

        run_model_comparison(self.args, landscape, masks, MODELS)