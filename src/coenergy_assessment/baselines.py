"""Modèles de référence simples, auxquels comparer les réseaux de neurones."""

from __future__ import annotations

import torch

from .data import SimulationData


def mean_baseline(y_train: torch.Tensor, n: int) -> torch.Tensor:
    """Prédit la moyenne de l'entraînement pour chaque indicateur.

    Args:
        y_train: indicateurs de l'entraînement (n_train, 6).
        n: nombre de prédictions à produire.
    """
    return y_train.mean(0).expand(n, -1).clone()


def summary_features(data: SimulationData, idx: torch.Tensor, every_min: int = 30) -> torch.Tensor:
    """Résumé de chaque simulation : élévation de T et énergie cumulée, échantillonnées.

    Args:
        data: simulations.
        idx: indices des simulations.
        every_min: pas d'échantillonnage, en minutes.

    Returns:
        Tenseur (n, 2 * L / every_min) : ``T(t) - T(0)`` puis l'énergie cumulée (kWh).
    """
    parts = []
    for a in range(0, len(idx), 20_000):                    # par morceaux, pour limiter la mémoire
        b = idx[a:a + 20_000]
        t, u = data.temperature(b), data.power(b)
        rise = (t - t[:, :1])[:, every_min - 1::every_min]
        energy = (u.cumsum(1) / 60 / 1000)[:, every_min - 1::every_min]
        parts.append(torch.cat([rise, energy], dim=1).cpu())
    return torch.cat(parts)


class RidgeBaseline:
    """Régression linéaire régularisée sur un résumé de la simulation.

    Elle indique ce qu'un modèle linéaire tire de l'élévation de température et de
    l'énergie injectée : c'est le niveau qu'un réseau de neurones doit dépasser pour se
    justifier. Comme elle est linéaire et que HLC = HLC_mit + HLC_ext dans les données,
    ses prédictions respectent exactement la contrainte de somme.

    Args:
        alpha: coefficient de régularisation (sur les caractéristiques standardisées).
    """

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha
        self.mu: torch.Tensor | None = None
        self.sd: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None

    def _design(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.double() - self.mu) / self.sd
        return torch.cat([z, torch.ones(len(z), 1, dtype=torch.float64)], dim=1)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "RidgeBaseline":
        """Ajuste la régression. ``x`` : (n, d) ; ``y`` : (n, 6)."""
        self.mu, self.sd = x.double().mean(0), x.double().std(0).clamp(min=1e-8)
        a = self._design(x)
        reg = self.alpha * len(a) * torch.eye(a.shape[1], dtype=torch.float64)
        reg[-1, -1] = 0.0                                   # pas de pénalité sur le biais
        self.weights = torch.linalg.solve(a.T @ a + reg, a.T @ y.double())
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit les 6 indicateurs (n, 6)."""
        if self.weights is None:
            raise RuntimeError("RidgeBaseline.fit doit être appelé avant predict")
        return (self._design(x) @ self.weights).float()
