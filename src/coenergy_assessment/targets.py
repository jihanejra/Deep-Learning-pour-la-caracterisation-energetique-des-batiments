"""Codages des cibles : passage des 6 indicateurs physiques aux 5 sorties du réseau.

Les indicateurs obéissent à deux contraintes physiques :

* ``HLC = HLC_mit + HLC_ext`` (somme exacte des déperditions aux frontières) ;
* ``tau_n < tau_inf < tau_b`` (ordre des constantes de temps).

Dans les deux codages, le réseau ne prédit jamais HLC : il est reconstruit comme la somme
de ses deux composantes, donc la première contrainte est toujours respectée. Le codage
``contraint`` garantit en plus l'ordre des constantes de temps.
"""

from __future__ import annotations

import torch

ENCODINGS: tuple[str, ...] = ("contraint", "direct")
N_OUTPUTS: int = 5


class TargetCodec:
    """Encode, standardise et décode les cibles.

    * ``contraint`` : ``HLC_mit, HLC_ext, log tau_n, log(tau_inf - tau_n), log(tau_b - tau_inf)``.
      Les écarts étant des exponentielles, donc positifs, l'ordre des tau est garanti.
    * ``direct`` : ``HLC_mit, HLC_ext, tau_b, tau_n, tau_inf``. L'ordre des tau n'est pas garanti.

    Args:
        encoding: ``"contraint"`` ou ``"direct"``.
        targets: noms des 6 indicateurs, dans l'ordre des colonnes de Y.
    """

    def __init__(self, encoding: str, targets: list[str]) -> None:
        if encoding not in ENCODINGS:
            raise ValueError(f"Codage inconnu : {encoding!r} (attendu : {ENCODINGS})")
        self.encoding = encoding
        self.targets = list(targets)
        self._col = {name: i for i, name in enumerate(self.targets)}
        self.mu: torch.Tensor | None = None
        self.sd: torch.Tensor | None = None

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Indicateurs physiques (n, 6) -> sorties non standardisées (n, 5)."""
        c = self._col
        mit, ext = y[:, c["HLC_mit"]], y[:, c["HLC_ext"]]
        tb, tn, ti = y[:, c["tau_b"]], y[:, c["tau_n"]], y[:, c["tau_inf"]]
        if self.encoding == "contraint":
            return torch.stack([mit, ext, tn.log(), (ti - tn).log(), (tb - ti).log()], dim=1)
        return torch.stack([mit, ext, tb, tn, ti], dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Sorties non standardisées (n, 5) -> indicateurs physiques (n, 6)."""
        c = self._col
        y = torch.empty(len(z), 6, dtype=z.dtype, device=z.device)
        y[:, c["HLC_mit"]], y[:, c["HLC_ext"]] = z[:, 0], z[:, 1]
        y[:, c["HLC"]] = z[:, 0] + z[:, 1]
        if self.encoding == "contraint":
            tn = z[:, 2].exp()
            ti = tn + z[:, 3].exp()
            tb = ti + z[:, 4].exp()
        else:
            tb, tn, ti = z[:, 2], z[:, 3], z[:, 4]
        y[:, c["tau_b"]], y[:, c["tau_n"]], y[:, c["tau_inf"]] = tb, tn, ti
        return y

    def fit(self, y_train: torch.Tensor) -> "TargetCodec":
        """Calcule la moyenne et l'écart-type des sorties sur l'ensemble d'entraînement."""
        z = self.encode(y_train.double())
        self.mu, self.sd = z.mean(0).float(), z.std(0).float()
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Indicateurs physiques (n, 6) -> cibles standardisées du réseau (n, 5)."""
        self._check_fitted()
        return ((self.encode(y.double()).float() - self.mu) / self.sd)

    def inverse(self, z_std: torch.Tensor) -> torch.Tensor:
        """Sorties standardisées du réseau (n, 5) -> indicateurs physiques (n, 6)."""
        self._check_fitted()
        z = z_std.double().cpu() * self.sd.double() + self.mu.double()
        return self.decode(z).float()

    def _check_fitted(self) -> None:
        if self.mu is None or self.sd is None:
            raise RuntimeError("TargetCodec.fit doit être appelé avant transform / inverse")


def constraint_violations(y: torch.Tensor, targets: list[str], atol: float = 1e-3) -> dict[str, float]:
    """Part des prédictions qui violent chaque contrainte physique.

    Args:
        y: indicateurs (n, 6).
        targets: noms des colonnes de ``y``.
        atol: tolérance absolue sur la somme des HLC (W/K).

    Returns:
        ``{"somme_HLC", "ordre_tau", "valeurs_negatives"}`` : parts entre 0 et 1.
    """
    c = {name: i for i, name in enumerate(targets)}
    somme = (y[:, c["HLC"]] - y[:, c["HLC_mit"]] - y[:, c["HLC_ext"]]).abs() > atol
    ordre = ~((y[:, c["tau_n"]] < y[:, c["tau_inf"]]) & (y[:, c["tau_inf"]] < y[:, c["tau_b"]]))
    negatif = (y <= 0).any(dim=1)
    return {"somme_HLC": somme.float().mean().item(), "ordre_tau": ordre.float().mean().item(),
            "valeurs_negatives": negatif.float().mean().item()}
