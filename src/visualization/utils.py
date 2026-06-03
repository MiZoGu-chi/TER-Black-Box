from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt

K_COLORS = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}

_RESULTS_DIR = "results/plots"


def save_plot(base_name, *tags, subdir=None):
    plot_dir = Path(_RESULTS_DIR) / subdir if subdir else Path(_RESULTS_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    filename = base_name
    for tag in tags:
        if tag:
            filename += f"_{tag}"

    path = plot_dir / f"{filename}.png"
    counter = 1
    while path.exists():
        path = plot_dir / f"{filename}_{counter}.png"
        counter += 1

    plt.savefig(path)
    print(f"Graphique sauvegardé : {path}")


@contextmanager
def figure(filename, *tags, subdir=None):
    try:
        yield
    finally:
        save_plot(filename, *tags, subdir=subdir)
        plt.close()


def save_df_as_image(df, filename="interaction_metrics", subdir=None):
    with figure(filename, subdir=subdir):
        fig, ax = plt.subplots(figsize=(10, 0.5 * (len(df) + 1)))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns,
                         rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))
