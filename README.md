## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## CLI

Point d'entrée unique : `main.py`, avec un sous-mode par commande.

```bash
python3 main.py <commande> [options]
```

### `optimize` — lancer des runs d'optimisation

Lance la boucle d'optimisation guidée par modèle de substitution sur une ou plusieurs instances de NK-landscape, et écrit un fichier de résultat par run.

```bash
python3 main.py optimize --N 64 --K 1 2 --D 2 --budget 10000 \
    --surrogate lasso_lars --algo tabu --alpha 0.00001 \
    --tabu_factor 2 --duplicate_strategy random
```

Options principales (description reprise du `--help`) :

- `--D` : ordre de troncature du modèle de substitution de Walsh (défaut 2)
- `--N` : paramètre N du NK-landscape, tout (choix `20, 64, 128, 256`, défaut `64`)
- `--K` : paramètre K du NK-landscape (choix `1, 2, 4, 8`, défaut `1 2`)
- `--budget` : budget boîte noire d'appelle à la fonction objectif (défaut 1000)
- `--initial_pop_size` : taille de la population intiale sur laquelle on entraine le modéle avant de commencer l'optimisation guidée (défaut 20)
- `--surrogate` : modèle de regression à utiliser (défaut `lasso_lars`, voir `src/core/models.py`)
- `--algo` : algorithme à utiliser pour l'optimisation guidée (défaut `tabu`, choix `hc_fi, hc_bi, tabu, drils_hc_bi, drils_hc_fi, drils_tabu`)
- `--start` : stratégie de départ utiliser à chaque itération de la boucle d'optimisation guidée (défaut `rd`)
- `--alpha` : terme de régularisation L1 (défaut 0.00001)
- `--perturbation_strength` : this (meta-)parameter is called the strength of the perturbation : drils perturbs the first parent by randomly flipping perturbation_strength * n bits (défaut 0.05)
- `--tabu_factor` : facteur d'exploration de l'aglorithme tabou (N * facteur d'exploration) (défaut 2)
- `--drils_factor` : facteur d'exploration de l'aglorithme DRILS (N * facteur d'exploration) (défaut 2)
- `--duplicate_strategy` : stratégie utiliser lorsque l'on tombe sur une solution déjà évaluté pendant la boucle d'optimisation guidée (défaut `random`, choix `ignore, random, skip`)
- `--instances` : pour tester des instances spécifiques (réduire le nombre d'instances ne réduit pas le temps d'exécution tant que n_jobs > n_instances) — exclusif avec `--n_instances` (défaut : toutes, 0-9)
- `--n_instances` : nombre d'instances (défaut 10)
- `--seeds` : seeds spécifiques — exclusif avec `--n_seeds`
- `--n_seeds` : nombre de seeds (défaut 10, donc 10 instances × 10 seeds = 100 runs par config)


### `rank` — classer les configurations par score

Agrège tous les fichiers de résultats par config `(N, K, algo, alpha, tabu_factor, drils_factor, budget)` et
affiche un tableau trié + génère les graphiques suivante :
- heatmap
- ...

```bash
python3 main.py rank results/opti_output/64
```

### `analyze` — ...

Reprend un ou plusieurs fichiers/dossiers de résultats et 
génère les graphes suivantes : convergence, de fidélité du modèle et de matrice de confusion
(vrais/faux positifs sur les coefficients appris).

```bash
python3 main.py analyze results/opti_output/64/2/1/lasso_lars/1e-05/tabu
```

### `evaluate` — évaluer la fidélité d'un modèle après l'optimisation guidée

Recharge le modèle appris (coefficients sauvegardés ds le fichier d'output) et le landscape réel associés à une instance, génère de nouveaux échantillons de test et mesure l'écart prédiction/réalité, gerner les graphes suivants :
///

```bash
python3 main.py evaluate results/opti_output/64/2/1/lasso_lars/1e-05/tabu
```

- `path` : chemin vers le(s) dossier(s) ou le(s) fichier(s) d'output de l'optimisation à analyser
- `--samples` : nombre de solution de test à générer et tester sur chaque fichiers d'ouput de l'optimisation (défaut 1000)
- `--sampler` : choix du sampler, bon en réalité on va prendre whitebox tout le temps.. (défaut `whitebox`)

### `compare` — comparer plusieurs modèles de régression sur une instance

Entraîne et compare plusieurs surrogates (`--models`) sur une seule instance
de NK-landscape (pas de boucle d'optimisation, juste un banc d'essai de
régression).

```bash
python3 main.py compare 64 1 --models lasso_lars lars cg_ridge
```

- `N` : paramètre N du NK-landscape
- `K` : paramètre K du NK-landscape
- `--instance` : indice de l'instance de NK-landscape à charger (0-9) (défaut 0)
- `--D` : ordre de troncature du modèle de substitution de Walsh (défaut 2)
- `--n_train` : nombre de solutions générées pour entraîner les modèles de régression (défaut 400)
- `--n_test` : nombre de solutions générées pour évaluer les modèles de régression (défaut 1000)
- `--models` : modèles de régression à comparer (défaut `lasso_lars`)
- `--loss_function` : fonction de perte utilisée pour comparer les modèles (défaut `mae`, choix `mae, mse`)

Pour `rank`, `analyze` et `evaluate`, l'argument `path` accepte aussi bien un
fichier `.txt` qu'un dossier — dans ce cas tous les `opti_*.txt` sous ce
dossier sont récupérés récursivement (`rglob`). Passer un dossier trop haut
dans la hiérarchie (ex: plusieurs runs de la même config) mélange les runs
dans la même moyenne.

## Lancer sur le cluster (SLURM / nautilus)

`run_expe.sh` est le wrapper `sbatch` (détecte le cluster via `hostname` et
adapte les options SLURM). Il est appelé par les scripts `run_*_seq_expe.sh`
à la racine, qui bouclent sur (alpha, N, K, instance, seed) et soumettent un
job par combinaison — un job = un seul `--instances X --seeds Y`. Variables
d'environnement lues par `run_expe.sh` : `SLURM_QOS` (défaut `short`) et
`SLURM_TIME` (défaut `1-00:00:00`).

```bash
bash run_lasso_lars_tabu_seq_expe.sh
```

## Hiérarchie des fichiers

### Instances (`instances/`)

```
instances/{N}/{K}/nk_{N}_{K}_{num}.txt       # instance num (0-9)
instances/{N}/{K}/expected_local_optima.json # optima locaux moyens de référence
```

Pré-générées et versionnées dans git (pas de commande pour les régénérer à
ce jour).

### Résultats d'optimisation (`results/opti_output/`)

```
results/opti_output/
  {N}/{D}/{K}/{surrogate}/{alpha}/{algo}/[{tabu_factor|drils_factor}]/
    run_{timestamp}_{job_id}/
      opti_{N}_{K}_{num}_{seed}_{algo}_{surrogate}.txt
```

- Le sous-dossier `{tabu_factor}` (ou `{drils_factor}`) n'existe que si
  `algo` contient `tabu` (resp. `drils`) — `hc_fi`/`hc_bi` s'arrêtent au
  niveau `{algo}`.
- `{alpha}` n'apparaît que si le surrogate a un paramètre `alpha` (cas des
  modèles lasso).
- Un dossier `run_{timestamp}_{job_id}` est créé à **chaque exécution** de
  `optimize` (un par job SLURM). Un fichier `opti_*.txt` correspond à une
  seule paire (instance `num`, `seed`). `job_id` vaut `local` hors SLURM.
- Comme `rank`/`analyze`/`evaluate` lisent tous les `opti_*.txt` sous le
  chemin passé, relancer une config crée un nouveau dossier `run_*`
  supplémentaire plutôt que d'écraser l'ancien — bien pointer vers le run
  voulu (ou nettoyer les anciens) pour ne pas mélanger plusieurs runs d'une
  même config dans une même moyenne.

Contenu d'un fichier `opti_*.txt` (voir `write_optimization_result` dans
`src/utils/io_utils.py`) : une ligne d'en-tête (N, K, D, num, seed, algo,
surrogate, mean_local_optima, alpha, tabu_factor, duplicate_strategy,
drils_factor), puis 4 sections séparées par `#` : historique
fitness/budget, historique fidélité prédite/réelle, meilleure solution
trouvée, coefficients du modèle appris.

### Graphiques (`results/plots/`)

```
results/plots/
  rank/{N}/score_vs_alpha_N{N}.png
  rank/{N}/heatmap_alpha_tabu_N{N}.png
  benchmark/{N}/{K}/...                 # sorties de `compare`
  ... (subdir + tags dépendent de la commande, voir src/visualization/)
```

Par défaut les fichiers ne sont pas écrasés (`overwrite=False` dans
`save_plot`) : un run répété ajoute `_1`, `_2`, ... au nom de fichier — sauf
pour `rank` qui force `overwrite=True`.
