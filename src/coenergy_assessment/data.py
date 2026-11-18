"""Préparation, chargement et découpage des simulations.

Les simulations brutes (batchs ``.pt``) sont converties une seule fois en tableaux
``.npy`` compacts : T (arrondie à 0,01 °C) est stockée en centièmes de degré et
U (0 ou 7000 W) en watts, toutes deux en ``int16``. Ce format prend deux fois moins
de place qu'en ``float32``, sans aucune perte.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .util_assessement import load_simulation_data

TARGETS: list[str] = ["HLC", "HLC_mit", "HLC_ext", "tau_b", "tau_n", "tau_inf"]
UNITS: dict[str, str] = {"HLC": "W/K", "HLC_mit": "W/K", "HLC_ext": "W/K",
                         "tau_b": "h", "tau_n": "h", "tau_inf": "h"}
FAMILIES: dict[str, list[str]] = {"HLC": ["HLC"], "Hb": ["HLC_mit", "HLC_ext"],
                                  "CDT": ["tau_b", "tau_n", "tau_inf"]}
DT_MIN: float = 1.0          # pas de temps des simulations (minutes)
T_SCALE: int = 100           # T est stockée en centièmes de degré


@dataclass
class SimulationData:
    """Jeu de simulations en mémoire, sous sa forme compacte.

    Attributes:
        T: température intérieure, ``int16`` (N, L), en centièmes de degré.
        U: puissance de chauffage, ``int16`` (N, L), en watts.
        Y: indicateurs cibles, ``float32`` (N, 6), dans l'ordre de ``targets``.
        targets: noms des 6 indicateurs.
    """

    T: torch.Tensor
    U: torch.Tensor
    Y: torch.Tensor
    targets: list[str]

    def __len__(self) -> int:
        return self.T.shape[0]

    @property
    def seq_len(self) -> int:
        """Nombre de pas de temps par simulation."""
        return self.T.shape[1]

    @property
    def hours(self) -> np.ndarray:
        """Axe du temps, en heures."""
        return np.arange(self.seq_len) * DT_MIN / 60

    def temperature(self, idx: torch.Tensor | slice) -> torch.Tensor:
        """Température en °C (``float32``) des simulations ``idx``."""
        return self.T[idx].float() / T_SCALE

    def power(self, idx: torch.Tensor | slice) -> torch.Tensor:
        """Puissance en W (``float32``) des simulations ``idx``."""
        return self.U[idx].float()

    def col(self, name: str) -> int:
        """Indice de la colonne de ``Y`` correspondant à l'indicateur ``name``."""
        return self.targets.index(name)

    def to(self, device: torch.device | str) -> "SimulationData":
        """Copie des séries T et U sur ``device`` (Y reste sur CPU pour l'évaluation)."""
        return SimulationData(self.T.to(device), self.U.to(device), self.Y, self.targets)


def prepare_data(raw_dir: str | Path, out_dir: str | Path) -> None:
    """Convertit tous les batchs bruts en ``.npy`` compacts. À exécuter une seule fois.

    Les batchs sont traités un par un : un seul est en mémoire à la fois.

    Args:
        raw_dir: dossier contenant ``batch_*.pt``.
        out_dir: dossier de sortie (créé si besoin).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = sorted(raw_dir.glob("batch_*.pt"))
    if not batches:
        raise FileNotFoundError(f"Aucun fichier batch_*.pt dans {raw_dir}")

    arrays: dict[str, np.ndarray] = {}
    for k, batch in enumerate(batches):
        # load_simulation_data lit un dossier entier : on lui présente un dossier ne
        # contenant que ce batch (lien physique, sans copie du fichier).
        with tempfile.TemporaryDirectory(dir=raw_dir.parent) as tmp:
            os.link(batch, Path(tmp) / batch.name)
            s = load_simulation_data(folder_path=Path(tmp))

        n, length = s.indoor_temperature.shape
        if k == 0:
            names = ["HLC", *s.Hb_labels, "tau_b", "tau_n", "tau_inf"]
            np.save(out_dir / "target_names.npy", np.array(names))
            total = n * len(batches)
            arrays = {
                "T": np.lib.format.open_memmap(out_dir / "T.npy", "w+", np.int16, (total, length)),
                "U": np.lib.format.open_memmap(out_dir / "U.npy", "w+", np.int16, (total, length)),
                "Y": np.lib.format.open_memmap(out_dir / "Y.npy", "w+", np.float32, (total, 6)),
            }

        rows = slice(k * n, (k + 1) * n)
        arrays["T"][rows] = torch.round(s.indoor_temperature * T_SCALE).to(torch.int16).numpy()
        arrays["U"][rows] = torch.round(s.indoor_U).to(torch.int16).numpy()
        arrays["Y"][rows] = torch.cat([s.HLC.unsqueeze(1), s.Hb, s.CDT], dim=1).numpy()
        del s
        print(f"Batch {k + 1}/{len(batches)} traité")

    for arr in arrays.values():
        arr.flush()


def load_data(data_dir: str | Path, n: int | None = None) -> SimulationData:
    """Charge les données préparées en mémoire, sous leur forme compacte.

    Args:
        data_dir: dossier contenant ``T.npy``, ``U.npy``, ``Y.npy`` et ``target_names.npy``.
        n: nombre de simulations à charger (les ``n`` premières) ; ``None`` = toutes.

    Returns:
        Les simulations (1,9 Go pour le million complet).
    """
    data_dir = Path(data_dir)
    rows = slice(None) if n is None else slice(0, n)
    T = np.load(data_dir / "T.npy", mmap_mode="r")
    U = np.load(data_dir / "U.npy", mmap_mode="r")
    Y = np.load(data_dir / "Y.npy", mmap_mode="r")
    targets = [str(t) for t in np.load(data_dir / "target_names.npy")]
    return SimulationData(
        T=torch.from_numpy(np.array(T[rows])),
        U=torch.from_numpy(np.array(U[rows])),
        Y=torch.from_numpy(np.array(Y[rows])),
        targets=targets,
    )


def excitation_groups(U: torch.Tensor, p_max: float) -> dict[str, torch.Tensor]:
    """Masques booléens (N,) des trois régimes d'excitation.

    Args:
        U: puissance de chauffage (N, L).
        p_max: puissance maximale (W).

    Returns:
        ``{"avec créneaux", "sans excitation", "toujours allumé"}`` : partition des simulations.
    """
    no_exc = U.amax(dim=1) == 0
    always_on = U.amin(dim=1) == p_max
    return {"avec créneaux": ~no_exc & ~always_on, "sans excitation": no_exc, "toujours allumé": always_on}


def make_splits(groups: dict[str, torch.Tensor], path: str | Path | None = None,
                seed: int = 42, frac_val: float = 0.1, frac_test: float = 0.1) -> dict[str, torch.Tensor]:
    """Découpage train / validation / test stratifié par régime d'excitation.

    Si ``path`` existe, le découpage enregistré est rechargé : toutes les expériences
    partagent ainsi exactement les mêmes ensembles.

    Args:
        groups: masques de ``excitation_groups`` (une partition des simulations).
        path: fichier ``.pt`` où lire ou écrire le découpage ; ``None`` = ne rien enregistrer.
        seed: graine du tirage.
        frac_val: part de la validation.
        frac_test: part du test.

    Returns:
        ``{"train", "val", "test"}`` : indices des simulations de chaque ensemble.
    """
    n_total = len(next(iter(groups.values())))
    if path is not None and Path(path).exists():
        splits = torch.load(path)
        if max(int(idx.max()) for idx in splits.values()) < n_total:
            return splits
        # découpage enregistré pour un jeu plus grand (essai sur un sous-ensemble) : on le restreint
        return {name: idx[idx < n_total] for name, idx in splits.items()}

    g = torch.Generator().manual_seed(seed)
    parts: dict[str, list[torch.Tensor]] = {"train": [], "val": [], "test": []}
    for mask in groups.values():
        idx = mask.nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_test, n_val = int(len(idx) * frac_test), int(len(idx) * frac_val)
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test:n_test + n_val])
        parts["train"].append(idx[n_test + n_val:])
    splits = {}
    for name, chunks in parts.items():
        idx = torch.cat(chunks)
        splits[name] = idx[torch.randperm(len(idx), generator=g)]   # sinon rangés par régime
    if path is not None:
        torch.save(splits, path)
    return splits
