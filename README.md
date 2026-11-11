# Estimation des paramètres thermiques de bâtiments par apprentissage profond

Estimation des indicateurs thermiques d'un bâtiment (coefficient de déperdition **HLC** et constantes de temps **τ**) à partir d'une courte série temporelle de température intérieure et de puissance de chauffage, à l'aide de réseaux de neurones séquentiels (**CNN-GRU**, **TCN**, **Transformer**).

## Contexte

Le jeu de données contient **1 000 000 de simulations** de bâtiments, tous distincts. Chaque simulation dure **8 heures, au pas d'une minute (480 pas)** et fournit :

| Entrée | Description |
|---|---|
| `T(t)` | Température intérieure (°C) |
| `U(t)` | Puissance de chauffage, signal tout-ou-rien (0 ou 7000 W) |

Le but est de prédire **6 indicateurs** par bâtiment :

| Cible | Unité | Signification |
|---|---|---|
| `HLC` | W/K | Coefficient de déperdition thermique global |
| `HLC_mit` | W/K | Part du HLC liée à la masse intérieure |
| `HLC_ext` | W/K | Part du HLC liée à l'enveloppe extérieure |
| `tau_b` | h | Constante de temps du bâtiment |
| `tau_n` | h | Constante de temps courte |
| `tau_inf` | h | Constante de temps intermédiaire |

Ces cibles obéissent à deux contraintes physiques : **HLC = HLC_mit + HLC_ext** et **τ_n < τ_∞ < τ_b**.

## Démarche

1. **Analyse exploratoire** (`rapport_EDA.ipynb`) : distribution et corrélation des cibles, vérification des contraintes physiques, analyse des excitations (signaux PRBS, simulations sans chauffage, chauffage permanent).
2. **Découpage stratifié** train / validation / test (80 / 10 / 10 %), qui conserve la proportion de chaque type d'excitation. Le découpage est enregistré dans `Data/processed/splits.pt` pour être réutilisé par tous les notebooks.
3. **Deux codages des cibles** sont comparés :
   - **contraint** : `HLC_mit`, `HLC_ext`, `log τ_n`, `log(τ_∞ − τ_n)`, `log(τ_b − τ_∞)`. Les deux contraintes sont respectées par construction.
   - **direct** : `HLC_mit`, `HLC_ext`, `τ_b`, `τ_n`, `τ_∞`. L'ordre des τ est vérifié après la prédiction.
4. **Modèles**, tous comparés à une baseline qui prédit la moyenne :
   - **CNN-GRU** : 3 blocs convolutifs suivis d'un GRU bidirectionnel à 2 couches. C'est l'architecture de l'article de référence.
   - **CNN-GRU amélioré** : 3 canaux d'entrée physiquement motivés s'ajoutent aux 2 canaux de base : `T − T_libre`, `T − T(0)` et le temps normalisé.
   - **TCN** : convolutions dilatées avec connexions résiduelles, dont le champ réceptif couvre toute la séquence.
   - **Transformer à patches** : 30 patches de 16 minutes, un token `[CLS]` et un encodage de position.
5. **Évaluation en unités physiques** : RMSE, MAE, P95, MAPE et R² pour chaque indicateur, calculés séparément sur les simulations avec et sans excitation.

## Structure du projet

```
coenergy-assessment/
├── Data/                          # non versionné (≈ 3 Go)
│   ├── raw/                       # batch_1.pt … batch_20.pt (simulations brutes)
│   └── processed/                 # T.npy, U.npy, Y.npy, target_names.npy, splits.pt
├── util_assessement.py            # chargement des batchs bruts (.pt)
├── data_utils.py                  # conversion en .npy compacts (int16) et chargement
├── rapport_EDA.ipynb              # analyse exploratoire, découpage, baseline
├── rapport_cnn_gru.ipynb          # CNN-GRU : codage contraint vs direct
├── rapport_cnn_gru_ameliore.ipynb # CNN-GRU : 2 canaux vs 5 canaux physiques
├── rapport_tcn.ipynb              # TCN : codage contraint vs direct
├── rapport_transformer.ipynb      # Transformer : codage contraint vs direct
├── src/coenergy_assessment/       # package Python
├── pyproject.toml
└── uv.lock
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python 3.12.

```bash
uv sync
```

## Données

Les données ne sont pas versionnées. Placez les batchs bruts dans `Data/raw/`, puis convertissez-les une seule fois au format compact :

```python
from data_utils import prepare_data, load_data

prepare_data("Data/raw", "Data/processed")      # une seule fois

T, U, Y, targets = load_data("Data/processed")  # ou load_data(..., idx=indices) pour un sous-ensemble
```

`T` est stocké en centièmes de degré et `U` en watts, tous deux en `int16`. Ce format prend deux fois moins de place qu'en `float32`, sans aucune perte. `load_data` lit les fichiers en *memory-map*, ce qui permet de charger un sous-ensemble sans saturer la mémoire.

## Utilisation

```bash
uv run jupyter lab
```

Exécutez d'abord `rapport_EDA.ipynb`, qui prépare les données et génère `splits.pt`, puis les notebooks des modèles.

> Les notebooks des modèles ont été exécutés sur Kaggle (GPU) : adaptez la variable `DATA` en tête de notebook pour une exécution locale.

## Auteur

Jihane — CI3A
