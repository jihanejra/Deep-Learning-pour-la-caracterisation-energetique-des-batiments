"""Descripteurs de l'excitation et construction des canaux d'entrée des réseaux."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import DT_MIN, SimulationData

N_CHANNELS: dict[str, int] = {"base": 2, "physique": 5}


def excitation_descriptors(u: torch.Tensor, dt_min: float = DT_MIN) -> dict[str, torch.Tensor]:
    """Résume chaque signal U(t) d'un lot de simulations en quelques grandeurs physiques.

    Args:
        u: puissance de chauffage (n, L), en watts.
        dt_min: pas de temps, en minutes.

    Returns:
        Un dictionnaire de tenseurs (n,) :

        * ``energie_kWh`` : énergie injectée ;
        * ``taux_allumage`` : part du temps où le chauffage est allumé (0 à 1) ;
        * ``n_commutations`` : nombre de changements d'état ;
        * ``n_creneaux`` : nombre de créneaux de chauffe ;
        * ``duree_creneau_h`` : durée moyenne d'un créneau de chauffe (NaN sans chauffe) ;
        * ``premier_allumage_h`` : instant du premier allumage (NaN sans chauffe) ;
        * ``instant_moyen_h`` : barycentre temporel de l'énergie (NaN sans chauffe).
    """
    u = u.float()
    length = u.shape[1]
    hours = torch.arange(length, device=u.device, dtype=torch.float32) * dt_min / 60
    on = u > 0
    du = u.diff(dim=1)
    n_on = on.sum(1).float()
    n_blocks = ((du > 0).sum(1) + on[:, 0]).float()
    nan = torch.full_like(n_on, float("nan"))
    heated = n_on > 0

    first_on = on.float().argmax(1).float() * dt_min / 60
    energy_wh = u.sum(1) * dt_min / 60
    return {
        "energie_kWh": energy_wh / 1000,
        "taux_allumage": n_on / length,
        "n_commutations": (du != 0).sum(1).float(),
        "n_creneaux": n_blocks,
        "duree_creneau_h": torch.where(heated, n_on * dt_min / 60 / n_blocks.clamp(min=1), nan),
        "premier_allumage_h": torch.where(heated, first_on, nan),
        "instant_moyen_h": torch.where(heated, (u @ hours) / u.sum(1).clamp(min=1), nan),
    }


def describe_excitations(U: torch.Tensor, chunk: int = 50_000) -> dict[str, torch.Tensor]:
    """Applique ``excitation_descriptors`` à tout le jeu de données, par morceaux."""
    parts = [excitation_descriptors(U[a:a + chunk]) for a in range(0, len(U), chunk)]
    return {name: torch.cat([p[name].cpu() for p in parts]) for name in parts[0]}


@dataclass
class InputScaler:
    """Normalisation des entrées et construction des canaux, par lot.

    Deux jeux de canaux sont proposés :

    * ``base`` (2 canaux) : température normalisée et ``U / P_max`` ;
    * ``physique`` (5 canaux) : les deux précédents, plus ``T - T_libre`` (écart à la
      réponse libre moyenne, approximation de la réponse au chauffage), ``T - T(0)``
      (élévation depuis le départ) et le temps normalisé (position dans la météo commune).

    Toutes les statistiques sont estimées sur l'ensemble d'entraînement uniquement.

    Attributes:
        t_mu: moyenne de T (°C).
        t_sd: écart-type de T (°C).
        t_free: réponse libre moyenne (L,), en °C.
        d_free_sd: écart-type de ``T - T_libre`` (°C).
        d_t0_sd: écart-type de ``T - T(0)`` (°C).
        p_max: puissance maximale (W).
    """

    t_mu: float
    t_sd: float
    t_free: torch.Tensor
    d_free_sd: float
    d_t0_sd: float
    p_max: float

    @classmethod
    def fit(cls, data: SimulationData, train_idx: torch.Tensor, no_exc: torch.Tensor,
            p_max: float, n_sample: int = 100_000) -> "InputScaler":
        """Estime les statistiques sur l'entraînement.

        Args:
            data: simulations.
            train_idx: indices de l'ensemble d'entraînement.
            no_exc: masque (N,) des simulations sans excitation.
            p_max: puissance maximale (W).
            n_sample: nombre de simulations d'entraînement utilisées pour les écarts-types.
        """
        sample = data.temperature(train_idx[:n_sample])
        free_idx = train_idx[no_exc[train_idx]]
        t_free = data.temperature(free_idx).mean(0)
        return cls(
            t_mu=sample.mean().item(),
            t_sd=sample.std().item(),
            t_free=t_free,
            d_free_sd=(sample - t_free).std().item(),
            d_t0_sd=(sample - sample[:, :1]).std().item(),
            p_max=float(p_max),
        )

    def to(self, device: torch.device | str) -> "InputScaler":
        """Copie de la normalisation sur ``device``."""
        return InputScaler(self.t_mu, self.t_sd, self.t_free.to(device),
                           self.d_free_sd, self.d_t0_sd, self.p_max)

    def build(self, t: torch.Tensor, u: torch.Tensor, inputs: str = "base",
              noise_c: float = 0.0) -> torch.Tensor:
        """Construit le tenseur d'entrée d'un lot.

        Args:
            t: température (B, L), en °C.
            u: puissance (B, L), en W.
            inputs: ``"base"`` ou ``"physique"``.
            noise_c: écart-type d'un bruit gaussien de capteur ajouté à T (°C) ; 0 = aucun.

        Returns:
            Tenseur (B, C, L), avec C = 2 ou 5.
        """
        if inputs not in N_CHANNELS:
            raise ValueError(f"Entrées inconnues : {inputs!r} (attendu : {tuple(N_CHANNELS)})")
        if noise_c > 0:
            t = t + noise_c * torch.randn_like(t)
        channels = [(t - self.t_mu) / self.t_sd, u / self.p_max]
        if inputs == "physique":
            time = torch.linspace(0, 1, t.shape[1], device=t.device).expand_as(t)
            channels += [(t - self.t_free) / self.d_free_sd,
                         (t - t[:, :1]) / self.d_t0_sd,
                         time]
        return torch.stack(channels, dim=1)
