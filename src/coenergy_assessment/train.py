"""Entraînement, sauvegarde, rechargement et prédiction.

Les séries T et U tiennent entièrement sur le GPU sous leur forme compacte (1,9 Go) :
les lots sont formés par simple indexation des tenseurs, sans ``DataLoader``. C'est
nettement plus rapide que de lire les simulations une par une.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .data import SimulationData
from .features import N_CHANNELS, InputScaler
from .models import build_model
from .targets import N_OUTPUTS, TargetCodec


@dataclass
class Experiment:
    """Description complète d'une expérience : tout ce qu'il faut pour la reproduire.

    Attributes:
        name: identifiant, utilisé pour les fichiers sauvegardés.
        model: ``"cnn_gru"`` ou ``"transformer"``.
        inputs: ``"base"`` (2 canaux) ou ``"physique"`` (5 canaux).
        encoding: codage des cibles, ``"contraint"`` ou ``"direct"``.
        epochs: nombre d'époques.
        batch_size: taille des lots.
        lr: taux d'apprentissage maximal (planification one-cycle).
        weight_decay: régularisation L2 découplée (AdamW).
        loss: ``"mse"`` ou ``"huber"``.
        noise_c: écart-type du bruit de capteur ajouté à T pendant l'entraînement (°C).
        grad_clip: norme maximale du gradient.
        seed: graine aléatoire.
        n_train: nombre de simulations d'entraînement utilisées ; ``None`` = toutes.
        amp: précision mixte sur GPU.
    """

    name: str
    model: str
    inputs: str = "base"
    encoding: str = "contraint"
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"
    noise_c: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    n_train: int | None = None
    amp: bool = True


@dataclass
class RunResult:
    """Résultat d'une expérience entraînée ou rechargée."""

    experiment: Experiment
    model: nn.Module
    codec: TargetCodec
    history: list[dict[str, float]]
    best_epoch: int
    train_seconds: float
    n_parameters: int


def _loss_fn(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Perte inconnue : {name!r} (attendu : 'mse' ou 'huber')")


def _checkpoint_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / f"{name}.pt"


@torch.no_grad()
def predict(model: nn.Module, data: SimulationData, idx: torch.Tensor, scaler: InputScaler,
            codec: TargetCodec, inputs: str, noise_c: float = 0.0, batch_size: int = 2048,
            seed: int = 0) -> torch.Tensor:
    """Prédit les 6 indicateurs, en unités physiques, pour les simulations ``idx``.

    Args:
        model: réseau entraîné.
        data: simulations, sur le même appareil que le modèle.
        idx: indices des simulations (l'ordre est conservé).
        scaler: normalisation des entrées, sur le même appareil.
        codec: codage des cibles (déjà ajusté).
        inputs: jeu de canaux, ``"base"`` ou ``"physique"``.
        noise_c: bruit de capteur ajouté à T (°C), pour les tests de robustesse.
        batch_size: taille des lots.
        seed: graine du bruit, pour un résultat reproductible.

    Returns:
        Prédictions (n, 6) sur CPU.
    """
    device = data.T.device
    model.eval()
    if noise_c > 0:
        torch.manual_seed(seed)
    idx = idx.to(device)
    outputs = []
    for a in range(0, len(idx), batch_size):
        b = idx[a:a + batch_size]
        x = scaler.build(data.temperature(b), data.power(b), inputs, noise_c)
        outputs.append(model(x).float().cpu())
    return codec.inverse(torch.cat(outputs))


def run_experiment(exp: Experiment, data: SimulationData, splits: dict[str, torch.Tensor],
                   scaler: InputScaler, out_dir: str | Path, retrain: bool = False,
                   verbose: bool = True) -> RunResult:
    """Entraîne une expérience, ou la recharge si elle a déjà été sauvegardée.

    Le meilleur modèle au sens de la perte de validation est conservé. Un fichier
    ``<name>.pt`` (poids, configuration, statistiques des cibles, historique) et un
    fichier ``<name>_history.json`` sont écrits dans ``out_dir``.

    Args:
        exp: description de l'expérience.
        data: simulations, déjà placées sur l'appareil de calcul.
        splits: indices train / val / test.
        scaler: normalisation des entrées, sur le même appareil que ``data``.
        out_dir: dossier des sauvegardes.
        retrain: ``True`` pour réentraîner même si une sauvegarde existe.
        verbose: affiche une ligne par époque.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(out_dir, exp.name)
    if path.exists() and not retrain:
        return load_experiment(path, data, device=data.T.device)

    device = data.T.device
    use_amp = exp.amp and device.type == "cuda"
    torch.manual_seed(exp.seed)

    train_idx = splits["train"] if exp.n_train is None else splits["train"][:exp.n_train]
    val_idx = splits["val"].to(device)
    codec = TargetCodec(exp.encoding, data.targets).fit(data.Y[train_idx])
    z_std = codec.transform(data.Y).to(device)              # cibles standardisées de tout le jeu
    train_idx = train_idx.to(device)

    model = build_model(exp.model, N_CHANNELS[exp.inputs], seq_len=data.seq_len, n_out=N_OUTPUTS).to(device)
    criterion = _loss_fn(exp.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    steps_per_epoch = len(train_idx) // exp.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=exp.lr, epochs=exp.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.1)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val, best_state, best_epoch = float("inf"), None, 0
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, exp.epochs + 1):
        t0 = time.time()
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        train_loss = torch.zeros((), device=device)
        for step in range(steps_per_epoch):
            b = perm[step * exp.batch_size:(step + 1) * exp.batch_size]
            x = scaler.build(data.temperature(b), data.power(b), exp.inputs, exp.noise_c)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                pred = model(x)
            loss = criterion(pred.float(), z_std[b])
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), exp.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            train_loss += loss.detach()
        train_loss_value = (train_loss / steps_per_epoch).item()

        # validation : même perte, sans bruit, sur des simulations jamais vues
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for a in range(0, len(val_idx), 2048):
                b = val_idx[a:a + 2048]
                x = scaler.build(data.temperature(b), data.power(b), exp.inputs)
                with torch.autocast(device.type, enabled=use_amp):
                    pred = model(x)
                val_loss += criterion(pred.float(), z_std[b]).item() * len(b)
                n_val += len(b)
        val_loss /= n_val

        history.append({"epoch": epoch, "train": train_loss_value, "val": val_loss,
                        "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t0})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[{exp.name}] époque {epoch:3d}/{exp.epochs} | train {train_loss_value:.4f} | "
                  f"val {val_loss:.4f} | lr {history[-1]['lr']:.1e} | {history[-1]['seconds']:.0f} s")

    assert best_state is not None
    model.load_state_dict(best_state)
    seconds = time.time() - start
    torch.save({"experiment": asdict(exp), "state_dict": best_state, "codec_mu": codec.mu,
                "codec_sd": codec.sd, "history": history, "best_epoch": best_epoch,
                "train_seconds": seconds}, path)
    (out_dir / f"{exp.name}_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return RunResult(exp, model, codec, history, best_epoch, seconds,
                     sum(p.numel() for p in model.parameters()))


def load_experiment(path: str | Path, data: SimulationData,
                    device: torch.device | str | None = None) -> RunResult:
    """Recharge une expérience sauvegardée par ``run_experiment``.

    Args:
        path: fichier ``<name>.pt``.
        data: simulations (pour la longueur des séquences et le nom des cibles).
        device: appareil du modèle ; par défaut celui de ``data``.
    """
    device = torch.device(device) if device is not None else data.T.device
    ckpt: dict[str, Any] = torch.load(path, map_location=device)
    exp = Experiment(**ckpt["experiment"])
    codec = TargetCodec(exp.encoding, data.targets)
    codec.mu, codec.sd = ckpt["codec_mu"].cpu(), ckpt["codec_sd"].cpu()
    model = build_model(exp.model, N_CHANNELS[exp.inputs], seq_len=data.seq_len, n_out=N_OUTPUTS).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return RunResult(exp, model, codec, ckpt["history"], ckpt["best_epoch"], ckpt["train_seconds"],
                     sum(p.numel() for p in model.parameters()))
