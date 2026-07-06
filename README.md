## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## CLI

Point d'entrée unique : `main.py`, avec un sous-mode par commande.

```bash
python3 main.py <commande> [options]
```

### `optimize`, lancer des runs d'optimisation

Lance la boucle d'optimisation guidée par modèle de substitution sur une ou plusieurs instances de NK-landscape, et écrit un fichier de résultat par run.

```bash
python3 main.py optimize --N <N...> --K <K...> --D <D> --budget <budget> --surrogate <surrogate> --algo <algo> --alpha <alpha> --tabu_factor <tabu_factor> --drils_factor <drils_factor> --duplicate_strategy <duplicate_strategy>
```

Une invocation directe comme celle-ci tourne en local (utile pour tester
rapidement une config). Pour les runs de fond (N=64, tous K/instances/seeds),
voir [Lancer sur le cluster](#lancer-sur-le-cluster-slurm--nautilus) plus bas.

Options principales (description reprise du `--help`) :

- `--D` : ordre de troncature du modèle de substitution de Walsh 
- `--N` : paramètre N du NK-landscape, tout 
- `--K` : paramètre K du NK-landscape 
- `--budget` : budget boîte noire d'appelle à la fonction objectif
- `--initial_pop_size` : taille de la population intiale sur laquelle on entraine le modéle avant de commencer l'optimisation guidée (défaut 20)
- `--surrogate` : modèle de regression à utiliser (défaut `lasso_lars`, voir `src/core/models.py`)
- `--algo` : algorithme à utiliser pour l'optimisation guidée (défaut `tabu`, choix `hc_fi, hc_bi, tabu, drils_hc_bi, drils_hc_fi, drils_tabu`)
- `--start` : stratégie de départ utiliser à chaque itération de la boucle d'optimisation guidée (défaut `rd`)
- `--alpha` : terme de régularisation L1 (défaut 0.00001)
- `--perturbation_strength` : force de la perturbation de DRILS (proportion de bits retournés sur le premier parent) (défaut 0.05)
- `--tabu_factor` : facteur d'exploration de l'aglorithme tabou (N * facteur d'exploration) (défaut 2)
- `--drils_factor` : facteur d'exploration de l'aglorithme DRILS (N * facteur d'exploration) (défaut 2)
- `--duplicate_strategy` : stratégie utiliser lorsque l'on tombe sur une solution déjà évaluté pendant la boucle d'optimisation guidée (défaut `random`, choix `ignore, random, skip`)
- `--instances` : pour tester des instances spécifiques (réduire le nombre d'instances ne réduit pas le temps d'exécution tant que n_jobs > n_instances), exclusif avec `--n_instances` (défaut : toutes, 0-9)
- `--n_instances` : nombre d'instances (défaut 10)
- `--seeds` : seeds spécifiques, exclusif avec `--n_seeds`
- `--n_seeds` : nombre de seeds (défaut 10, donc 10 instances × 10 seeds = 100 runs par config)


### `rank`, classer les configurations par score

Agrège tous les fichiers de résultats par config `(N, K, algo, alpha, tabu_factor, drils_factor, budget)` et
affiche un tableau trié + génère, sur l'ensemble des résultats (toutes configs confondues) :

- **Score en fonction d'alpha** (`rank/{N}/score_vs_alpha_N{N}.png`) : score
  agrégé (moyenne ± écart-type) en fonction de `alpha` (échelle `symlog`),
  une courbe par `K`. Attention, le regroupement se fait uniquement par
  `alpha` (toutes les autres dimensions confondues) : si plusieurs
  `algo`/`tabu_factor`/`drils_factor`/`budget` partagent la même valeur
  d'`alpha`, ils sont agrégés ensemble sur ce graphique, potentiellement
  trop de choses différentes moyennées entre elles.
- **Heatmap alpha × tabu_factor** (`rank/{N}/heatmap_alpha_tabu_N{N}.png`) :
  grille `alpha` (x) × `tabu_factor` (y) du score, une sous-figure par `K`
  (valeurs affichées en texte dans les cellules si la grille contient au
  plus 80 cellules).

```bash
python3 main.py rank results/opti_output/64
```

### `analyze`, analyser la convergence et la fidélité d'un ensemble de runs

Reprend un ou plusieurs fichiers/dossiers de résultats, affiche pour chaque
`K` le score agrégé, l'écart-type et l'amélioration par rapport à la moyenne
des optima locaux, et génère (le nom des fichiers reflète les métadonnées du
premier fichier traité (N, surrogate, algo, alpha, tabu_factor), donc trompeur
si on mélange plusieurs configs différentes) :

- **Convergence** (`optimization_convergence_trajectories_{tag}.png`) :
  fitness moyenne (les trajectoires sont ré-échantillonnées en escalier sur
  une grille commune de nombre d'évaluations) en fonction du nombre
  d'évaluations, une courbe par `K`, avec une ligne pointillée horizontale =
  moyenne des optima locaux.
- **Fidélité pendant l'optimisation** (`model_training_fidelity_{tag}.png`) :
  un sous-graphe par `K`, nuage de points fitness prédite vs réelle observé
  pendant l'optimisation (couleur = nombre d'évaluations au moment de la
  mesure).
- **Matrice de confusion des interactions** (`confusion_distribution_{tag}.png`) :
  un sous-graphe par `K`, boîtes à moustaches du nombre de vrais/faux
  positifs (TP/FP) par ordre d'interaction (1 à D) sur les coefficients
  appris, étoile rouge = nombre réel médian d'interactions de cet ordre dans
  le NK-landscape.
- **Distribution de la fitness finale** (`boxplot_fitness_{tag}.png`) : boîte
  à moustaches de la fitness finale par `K`, étoile = moyenne des optima
  locaux.

```bash
python3 main.py analyze results/opti_output/64
```

### `evaluate`, évaluer la fidélité d'un modèle après l'optimisation guidée

Recharge le modèle appris (coefficients sauvegardés ds le fichier d'output) et le landscape réel associés à une instance, génère de nouveaux échantillons de test et mesure l'écart prédiction/réalité, génère les graphes suivants :

- **Fidélité globale** (`model_fidelity_{tag}.png`) : nuage de points
  fitness réelle vs prédite sur les nouveaux échantillons de test, une
  couleur par `K`.
- **Covariate shift** (`covariate_shift_{tag}.png`) : un sous-graphe par
  `K`, MAE moyenne (± écart-type en zone ombrée) en fonction de la distance
  de Hamming à la meilleure solution trouvée (`best_sol`).
- **Covariate scatter** (`covariate_scatter_K{K}_{tag}.png`) : un fichier
  par `K`, nuage de points fitness réelle vs prédite coloré par distance de
  Hamming à la meilleure solution trouvée.

```bash
python3 main.py evaluate results/opti_output/64
```

- `path` : chemin vers le(s) dossier(s) ou le(s) fichier(s) d'output de l'optimisation à analyser
- `--samples` : nombre de solution de test à générer et tester sur chaque fichiers d'ouput de l'optimisation (défaut 1000)
- `--sampler` : choix du sampler, bon en réalité on va prendre whitebox tout le temps.. (défaut `whitebox`)

### `compare`, comparer plusieurs modèles de régression sur une instance

Entraîne et compare plusieurs surrogates (`--models`) sur une seule instance
de NK-landscape (pas de boucle d'optimisation, juste un banc d'essai de
régression). Génère un graphe de l'erreur (MAE ou MSE) en fonction de la
taille de l'échantillon d'entraînement, une courbe par modèle comparé
(`benchmark/{N}/{K}/{N}_{K}_{num}_{D}_{loss_function}_{seed}_{models-tag}.png`).

```bash
python3 main.py compare <N> <K> --models <model...>
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
fichier `.txt` qu'un dossier, dans ce cas tous les `opti_*.txt` sous ce
dossier sont récupérés récursivement. Passer un dossier trop haut
dans la hiérarchie (ex: plusieurs runs de la même config) mélange les runs
dans la même moyenne.

## Lancer sur le cluster (SLURM / nautilus)

Seul `optimize` a besoin du cluster (c'est la seule commande coûteuse en
calcul) : `rank`, `analyze`, `evaluate` et `compare` s'exécutent toujours en
local, sur les fichiers de `results/opti_output/` une fois rapatriés depuis
le cluster (`scp`, pas d'automatisation pour l'instant).

```bash
scp -r Nautilus:~/TER-Black-Box/results/opti_output/{N}/{D}/{K}/{surrogate}/{alpha}/{algo}/[{drils_factor}]/[{tabu_factor}]/ /chemin/vers/TER-Black-Box/results/opti_output/{N}/{D}/{K}/{surrogate}/{alpha}/{algo}/[{drils_factor}]/[{tabu_factor}]
```cd

où `{N}/{D}/{K}/{surrogate}/{alpha}/{algo}/[{drils_factor}]/[{tabu_factor}]` suit
la [hiérarchie des résultats d'optimisation](#résultats-doptimisation-resultsopti_output)
ci-dessous (`{drils_factor}` n'existe que si `algo` contient `drils`,
`{tabu_factor}` que si `algo` contient `tabu` ; `drils_tabu` a donc les deux
niveaux imbriqués).

`slurm/run_expe.sh` est le wrapper `sbatch` (détecte le cluster via
`hostname` et adapte les options SLURM, nautilus ou stargate). Il est
appelé par les scripts `slurm/run_*_seq_expe.sh`, qui bouclent sur (alpha,
N, K, instance, seed) et soumettent un job par combinaison, un job = un
seul `--instances X --seeds Y`. Variables d'environnement lues par
`slurm/run_expe.sh` : `SLURM_QOS` (défaut `short`) et `SLURM_TIME` (défaut
`1-00:00:00`). Tous ces scripts s'exécutent depuis la racine du dépôt
(chemins relatifs vers `main.py`, `results/`, etc.).

```bash
bash slurm/run_lasso_lars_tabu_seq_expe.sh
```

`slurm/run_lasso_lars_tabu_expe.sh` (sans `_seq`) est différent : un seul job
SLURM qui parallélise toutes les instances/seeds en interne (joblib,
`CPUS_PER_TASK=10`) au lieu de soumettre un job par (instance, seed).

Il n'existe pour l'instant aucun script générique/paramétrable pour lancer
une expérience : chaque `slurm/run_*_expe.sh` est écrit à la main pour une
combinaison (algo, surrogate, alpha(s), N, K(s)) donnée. Pour tester une
nouvelle configuration, il faut écrire un nouveau script sur le modèle de
ceux déjà présents dans `slurm/`.

## Hiérarchie des fichiers

### Instances (`instances/`)

```
instances/{N}/{K}/nk_{N}_{K}_{num}.txt       # instance num (0-9)
```

Pré-générées et versionnées dans git (pas de commande pour les régénérer à
ce jour).

### Résultats d'optimisation (`results/opti_output/`)

```
results/opti_output/
  {N}/{D}/{K}/{surrogate}/{alpha}/{algo}/[{drils_factor}]/[{tabu_factor}]/
    run_{timestamp}_{job_id}/
      opti_{N}_{K}_{num}_{seed}_{algo}_{surrogate}.txt
```

- Le sous-dossier `{drils_factor}` n'existe que si `algo` contient `drils`,
  et `{tabu_factor}` que si `algo` contient `tabu` ; `hc_fi`/`hc_bi`
  s'arrêtent au niveau `{algo}`, `drils_hc_fi`/`drils_hc_bi` s'arrêtent à
  `{drils_factor}`, et `drils_tabu` a les deux niveaux imbriqués
  (`{drils_factor}/{tabu_factor}`).
- `{alpha}` n'apparaît que si le surrogate a un paramètre `alpha` (cas des
  modèles lasso).
- Un dossier `run_{timestamp}_{job_id}` est créé à **chaque exécution** de
  `optimize` (un par job SLURM). Un fichier `opti_*.txt` correspond à une
  seule paire (instance `num`, `seed`). `job_id` vaut `local` hors SLURM.
- Comme `rank`/`analyze`/`evaluate` lisent tous les `opti_*.txt` sous le
  chemin passé, relancer une config crée un nouveau dossier `run_*`
  supplémentaire plutôt que d'écraser l'ancien, bien pointer vers le run
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
  rank/{N}/score_vs_alpha_N{N}.png                     # `rank`
  rank/{N}/heatmap_alpha_tabu_N{N}.png                 # `rank`
  {N}/{K}/{surrogate}/[{alpha}]/{algo}/[{tabu_factor}]/
    optimization_convergence_trajectories_{tag}.png    # `analyze`
    model_training_fidelity_{tag}.png                  # `analyze`
    confusion_distribution_{tag}.png                   # `analyze`
    boxplot_fitness_{tag}.png                           # `analyze`
    model_fidelity_{tag}.png                            # `evaluate`
    covariate_shift_{tag}.png                           # `evaluate`
    covariate_scatter_K{K}_{tag}.png                    # `evaluate`
  benchmark/{N}/{K}/{N}_{K}_{num}_{D}_{loss_function}_{seed}_{models-tag}.png  # `compare`
```

`{tag}` correspond au chemin `{N}/{K}/{surrogate}/[{alpha}]/{algo}/[{tabu_factor}]`
ci-dessus avec les `/` remplacés par `_` ; pour `analyze`, `{K}` peut être
plusieurs valeurs jointes par `_` si les fichiers passés couvrent plusieurs
`K`. `[{alpha}]`/`[{tabu_factor}]` ne sont présents que dans les mêmes
conditions que dans la hiérarchie de `results/opti_output/` ci-dessus.

Par défaut les fichiers ne sont pas écrasés : un run répété ajoute `_1`,
`_2`, ... au nom de fichier, sauf pour `rank` qui écrase toujours le
précédent.
