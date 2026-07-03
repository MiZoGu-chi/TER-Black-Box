from .base import Command
from src.utils.ranking_report import build_ranking_dataframe
from src.visualization.ranking import plot_score_vs_alpha, plot_heatmap_alpha_tabu

class RankCommand(Command):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("path", type=str, nargs='+', help="Chemin vers le(s) dossier(s) ou le(s) fichier(s) d'output de l'optimisation à analyser")

    def execute(self):
        files = self.get_files()
        df = build_ranking_dataframe(files)

        for (N, K), group in df.groupby(['N', 'K'], sort=False):
            group = group.sort_values('score', ascending=False).reset_index(drop=True)
            group.index += 1

            print(f"\n{'='*60}")
            print(f"N={N}, K={K}")
            print(f"{'='*60}")

            display = group[['algo', 'alpha', 'tabu_factor', 'drils_factor', 'budget', 'score', 'std', 'n_runs', 'amelioration']].copy()
            display['alpha'] = display['alpha'].map(lambda x: f"{x:g}")
            display['amelioration'] = display['amelioration'].map(lambda x: f"{x:+.4f}" if x is not None else "N/A")
            print(display.to_string(float_format=lambda x: f"{x:.4f}"))

        plot_score_vs_alpha(df)
        plot_heatmap_alpha_tabu(df)
