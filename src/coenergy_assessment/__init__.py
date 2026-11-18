"""Estimation d'indicateurs thermiques de bâtiments à partir de séries temporelles simulées.

Modules :

* ``data`` : préparation, chargement et découpage des simulations ;
* ``features`` : descripteurs de l'excitation et canaux d'entrée des réseaux ;
* ``targets`` : codages des cibles et contraintes physiques ;
* ``models`` : CNN-GRU et Transformer à patches ;
* ``train`` : entraînement, sauvegarde et prédiction ;
* ``metrics`` : métriques par indicateur, par famille et par régime ;
* ``baselines`` : modèles de référence simples ;
* ``plots`` : figures du rapport.
"""

from .baselines import RidgeBaseline, mean_baseline, summary_features
from .data import (FAMILIES, TARGETS, UNITS, SimulationData, excitation_groups, load_data,
                   make_splits, prepare_data)
from .features import N_CHANNELS, InputScaler, describe_excitations, excitation_descriptors
from .metrics import family_summary, global_score, metrics_by_group, regression_metrics
from .models import CNNGRURegressor, PatchTransformerRegressor, build_model, count_parameters
from .targets import TargetCodec, constraint_violations
from .train import Experiment, RunResult, load_experiment, predict, run_experiment

__all__ = [
    "FAMILIES", "TARGETS", "UNITS", "N_CHANNELS",
    "SimulationData", "prepare_data", "load_data", "excitation_groups", "make_splits",
    "InputScaler", "excitation_descriptors", "describe_excitations",
    "TargetCodec", "constraint_violations",
    "CNNGRURegressor", "PatchTransformerRegressor", "build_model", "count_parameters",
    "Experiment", "RunResult", "run_experiment", "load_experiment", "predict",
    "regression_metrics", "metrics_by_group", "family_summary", "global_score",
    "mean_baseline", "RidgeBaseline", "summary_features",
]
