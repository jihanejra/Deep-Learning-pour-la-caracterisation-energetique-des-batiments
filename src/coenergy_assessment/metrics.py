"""Métriques d'évaluation, en unités physiques, par indicateur, par famille et par régime."""

from __future__ import annotations

import pandas as pd
import torch

from .data import FAMILIES

METRICS: tuple[str, ...] = ("RMSE", "MAE", "P95", "MAPE", "R2")


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, targets: list[str]) -> pd.DataFrame:
    """Métriques par indicateur.

    * ``RMSE``, ``MAE`` : erreurs quadratique et absolue moyennes (unité de l'indicateur) ;
    * ``P95`` : 95ᵉ centile de l'erreur absolue, le critère de fiabilité de l'article de référence ;
    * ``MAPE`` : erreur relative moyenne (%) ;
    * ``R2`` : coefficient de détermination.

    Args:
        y_true: valeurs de référence (n, 6).
        y_pred: prédictions (n, 6).
        targets: noms des colonnes.

    Returns:
        Un tableau indexé par indicateur, une colonne par métrique.
    """
    rows = {}
    for j, name in enumerate(targets):
        t, p = y_true[:, j].double(), y_pred[:, j].double()
        err = p - t
        ss_tot = (t - t.mean()).pow(2).sum()
        rows[name] = {
            "RMSE": err.pow(2).mean().sqrt().item(),
            "MAE": err.abs().mean().item(),
            "P95": err.abs().quantile(0.95).item() if len(err) <= 16_000_000 else float("nan"),
            "MAPE": 100 * (err / t).abs().mean().item(),
            "R2": (1 - err.pow(2).sum() / ss_tot).item(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")[list(METRICS)].rename_axis("indicateur")


def metrics_by_group(y_true: torch.Tensor, y_pred: torch.Tensor, targets: list[str],
                     groups: dict[str, torch.Tensor]) -> pd.DataFrame:
    """Métriques par indicateur, calculées sur l'ensemble puis sur chaque régime d'excitation.

    Args:
        y_true: valeurs de référence (n, 6).
        y_pred: prédictions (n, 6).
        targets: noms des colonnes.
        groups: masques booléens (n,) alignés sur ``y_true``.

    Returns:
        Un tableau indexé par (régime, indicateur). Le régime ``"tout"`` couvre l'ensemble.
    """
    frames = {"tout": regression_metrics(y_true, y_pred, targets)}
    for name, mask in groups.items():
        if mask.sum() > 1:
            frames[name] = regression_metrics(y_true[mask], y_pred[mask], targets)
    return pd.concat(frames, names=["régime"])


def family_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Moyenne des métriques par famille d'indicateurs (HLC, Hb, CDT).

    Args:
        table: sortie de ``regression_metrics`` (indexée par indicateur).
    """
    rows = {family: table.loc[names].mean() for family, names in FAMILIES.items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("famille")


def global_score(table: pd.DataFrame) -> float:
    """Score unique pour comparer les modèles : RMSE normalisée moyenne sur les 6 indicateurs.

    La RMSE de chaque indicateur est divisée par l'écart-type de l'indicateur
    (``sqrt(1 - R²)``) : 1 correspond à la prédiction de la moyenne, 0 à une prédiction parfaite.
    """
    return float((1 - table["R2"]).clip(lower=0).pow(0.5).mean())
