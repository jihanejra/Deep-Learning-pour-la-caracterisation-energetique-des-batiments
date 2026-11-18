# De la série temporelle aux indicateurs thermiques d'un bâtiment

Assessment de stage coEnergy. À partir de 8 heures de température intérieure `T(t)` et de puissance de chauffage `U(t)`, un réseau de neurones prédit six indicateurs thermiques intrinsèques d'un bâtiment.

Le rapport complet (physique du problème, analyse exploratoire, méthode, résultats, analyse des erreurs) est le notebook [`rapport.ipynb`](rapport.ipynb).

## Le problème

Le jeu de données contient 1 000 000 de simulations de la maison `quentin_house`. Chaque simulation est une variante thermique du même bâtiment (propriétés des parois tirées au hasard), soumise pendant 8 heures, au pas d'une minute, à un scénario de chauffage différent.

| Entrée | Description |
|---|---|
| `T(t)` | Température de l'air intérieur (°C) |
| `U(t)` | Puissance de chauffage, signal binaire pseudo-aléatoire (0 ou 7000 W) |

| Famille | Cible | Unité | Signification |
|---|---|---|---|
| HLC | `HLC` | W/K | Coefficient de déperdition thermique global |
| Hb | `HLC_ext` | W/K | Déperditions vers l'extérieur |
| Hb | `HLC_mit` | W/K | Déperditions vers les espaces mitoyens |
| CDT | `tau_b` | h | Décharge après saturation (inertie structurelle globale) |
| CDT | `tau_n` | h | Persistance de la fraîcheur nocturne (après 8 h de refroidissement) |
| CDT | `tau_inf` | h | Temps de séjour moyen (limite de `tau_n` pour une sollicitation infinie) |

Deux contraintes physiques relient les cibles : `HLC = HLC_ext + HLC_mit` et `tau_n < tau_inf < tau_b`.

## Démarche

1. **Analyse exploratoire** : distributions et corrélations des cibles, vérification des contraintes, nature des excitations, réponse libre, lien entre les entrées et les cibles.
2. **Découpage stratifié** par régime d'excitation : entraînement 80 %, validation 10 %, test 10 %.
3. **Codage des cibles** : le réseau prédit 5 valeurs et le HLC est reconstruit par somme. Le codage `contraint` garantit en plus l'ordre des constantes de temps ; il est comparé au codage `direct`.
4. **Entrées** : 2 canaux (`T` normalisée, `U / P_max`) ou 5 canaux physiquement motivés (ajout de `T − T_libre`, `T − T(0)` et du temps).
5. **Modèles PyTorch** : CNN-GRU (architecture de l'article de référence) et Transformer à patches, comparés à la moyenne et à une régression linéaire.
6. **Évaluation en unités physiques** : RMSE, MAE, P95, MAPE et R² par indicateur, par famille et par régime d'excitation ; analyse de l'erreur selon l'excitation ; robustesse au bruit de capteur.

## Résultats

Modèle retenu : **CNN-GRU à 5 canaux, codage contraint**. Métriques sur le test (99 999 simulations), en unités physiques :

| Indicateur | RMSE | Erreur relative moyenne | P95 de l'erreur absolue | R² |
|---|---|---|---|---|
| `HLC` | 23,5 W/K | 8,2 % | 46,1 W/K | 0,44 |
| `HLC_ext` | 12,3 W/K | 9,0 % | 24,0 W/K | 0,48 |
| `HLC_mit` | 22,3 W/K | 15,7 % | 43,4 W/K | 0,28 |
| `tau_b` | 9,4 h | 14,7 % | 18,6 h | 0,13 |
| `tau_n` | 1,7 h | 12,2 % | 3,4 h | 0,26 |
| `tau_inf` | 7,2 h | 17,8 % | 14,1 h | 0,10 |

- Les contraintes physiques sont respectées sur toutes les prédictions.
- Tous les réseaux battent la régression linéaire ; le CNN-GRU devance le Transformer.
- Le codage contraint ne coûte rien en précision ; les canaux physiques apportent un gain faible (0,2 à 1,1 % de RMSE).
- Les constantes de temps longues (`tau_b`, `tau_inf`) restent peu identifiables en 8 heures de mesure.
- Un modèle entraîné sans bruit ne résiste pas à un bruit de capteur de 0,25 °C ; un modèle entraîné avec ce bruit reste robuste.

## Structure du projet

```
coenergy-assessment/
├── rapport.ipynb                  # rapport exécutable, livré avec ses sorties
├── src/coenergy_assessment/       # package Python
│   ├── data.py                    # préparation, chargement, découpage
│   ├── features.py                # descripteurs de l'excitation, canaux d'entrée
│   ├── targets.py                 # codages des cibles, contraintes physiques
│   ├── models.py                  # CNN-GRU, Transformer
│   ├── train.py                   # entraînement, sauvegarde, prédiction
│   ├── metrics.py                 # métriques par indicateur, famille, régime
│   ├── baselines.py               # moyenne, régression linéaire
│   ├── plots.py                   # figures du rapport
│   └── util_assessement.py        # lecture des batchs bruts (fourni par coEnergy)
├── outputs/
│   ├── checkpoints/               # modèles entraînés
│   └── results/                   # tableaux de résultats (JSON, CSV)
├── notebooks/archive/             # notebooks de travail antérieurs (non maintenus)
├── Data/                          # non versionné
│   ├── raw/                       # batch_1.pt … batch_20.pt
│   └── processed/                 # T.npy, U.npy, Y.npy, target_names.npy, splits.pt
├── pyproject.toml
└── uv.lock
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python 3.12.

```bash
uv sync
```

## Données

Les données ne sont pas versionnées. Placez les batchs bruts dans `Data/raw/`, puis convertissez-les une seule fois :

```bash
uv run python -c "import coenergy_assessment as ca; ca.prepare_data('Data/raw', 'Data/processed')"
```

`T` est stockée en centièmes de degré et `U` en watts, en `int16` : 1,9 Go pour le million de simulations, sans perte.

## Exécution du rapport

Le rapport a été exécuté sur Kaggle (GPU). Le jeu de données Kaggle contient :

| Fichier | Rôle |
|---|---|
| `T.npy`, `U.npy`, `Y.npy`, `target_names.npy` | données prétraitées par `prepare_data` |
| `splits.pt` | découpage train / validation / test |
| `coenergy_assessment.zip` | le package (tout le code) |
| `checkpoints.zip` | les modèles entraînés (facultatif) |

Les chemins sont fixés dans la première cellule de code (`DATA`, `OUT`, `CKPT`). Un modèle déjà présent dans `CKPT` est rechargé au lieu d'être réentraîné : avec `checkpoints.zip`, le rapport s'exécute en une quinzaine de minutes ; sans lui, l'entraînement complet prend environ 1 h 15 sur un GPU T4.

En local, avec le jeu complet, il faut environ 8 Go de mémoire libre :

```bash
uv run jupyter lab rapport.ipynb
```

## Référence

R. Nassif, S. Guernouti, P. Stabat, M. Cohen. *Building-specific excitation optimization and simulation-driven machine learning for rapid in situ identification of the heat loss coefficient.* Prépublication SSRN 6928008.

