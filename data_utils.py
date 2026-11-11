"""Préparation des simulations dans un format compact sur disque, puis chargement.

T (arrondi à 0,01 °C) est stocké en centièmes de degré et U (0 ou 7000 W) en watts,
tous deux en int16 : deux fois moins de place qu'en float32, sans aucune perte.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from util_assessement import load_simulation_data


def prepare_data(raw_dir: str | Path, out_dir: str | Path) -> None:
    """À exécuter une seule fois : convertit tous les batchs en .npy compacts sur le disque.

    Les batchs sont traités un par un, donc un seul est en mémoire à la fois.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = sorted(p for p in raw_dir.iterdir() if not p.name.startswith("."))

    arrays = {}
    for k, batch in enumerate(batches):
        # load_simulation_data lit un dossier entier : on lui présente un dossier ne
        # contenant que ce batch (lien physique, sans copie du fichier)
        with tempfile.TemporaryDirectory(dir=raw_dir.parent) as tmp:
            os.link(batch, Path(tmp) / batch.name)
            s = load_simulation_data(folder_path=tmp)

        n, length = s.indoor_temperature.shape
        if k == 0:
            np.save(out_dir / "target_names.npy", np.array(["HLC", *s.Hb_labels, "tau_b", "tau_n", "tau_inf"]))
            shape = (n * len(batches),)
            arrays = {
                "T": np.lib.format.open_memmap(out_dir / "T.npy", "w+", np.int16, shape + (length,)),
                "U": np.lib.format.open_memmap(out_dir / "U.npy", "w+", np.int16, shape + (length,)),
                "Y": np.lib.format.open_memmap(out_dir / "Y.npy", "w+", np.float32, shape + (6,)),
            }

        rows = slice(k * n, (k + 1) * n)
        arrays["T"][rows] = torch.round(s.indoor_temperature * 100).to(torch.int16).numpy()
        arrays["U"][rows] = torch.round(s.indoor_U).to(torch.int16).numpy()
        arrays["Y"][rows] = torch.cat([s.HLC.unsqueeze(1), s.Hb, s.CDT], dim=1).numpy()
        del s
        print(f"Batch {k + 1}/{len(batches)} traité")

    for arr in arrays.values():
        arr.flush()


def load_data(out_dir: str | Path, idx=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Charge les données préparées en tenseurs float32 : T (°C), U (W), Y, noms des cibles.

    idx : indices des simulations à charger (None = toutes). Seules ces simulations sont
    lues sur le disque, ce qui permet de travailler par morceaux sans saturer la mémoire.
    """
    out_dir = Path(out_dir)
    T = np.load(out_dir / "T.npy", mmap_mode="r")
    U = np.load(out_dir / "U.npy", mmap_mode="r")
    Y = np.load(out_dir / "Y.npy", mmap_mode="r")
    idx = slice(None) if idx is None else np.sort(np.asarray(idx))

    return (
        torch.from_numpy(T[idx].astype(np.float32)) / 100,
        torch.from_numpy(U[idx].astype(np.float32)),
        torch.from_numpy(np.array(Y[idx])),
        list(np.load(out_dir / "target_names.npy")),
    )
