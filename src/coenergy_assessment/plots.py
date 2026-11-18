"""Figures du rapport. Chaque fonction renvoie la figure matplotlib qu'elle crée."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from .data import UNITS

# Palette catégorielle (ordre fixe) et teintes associées
SERIES: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
SEQUENTIAL = LinearSegmentedColormap.from_list("bleu", ["#cde2fb", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("bleu_orange", ["#2a78d6", "#f0efec", "#eb6834"])


def set_style() -> None:
    """Applique le style commun à toutes les figures du rapport."""
    plt.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def _label(name: str) -> str:
    return f"{name} ({UNITS[name]})" if name in UNITS else name


def plot_trajectories(hours: np.ndarray, t: torch.Tensor, u: torch.Tensor, titles: list[str],
                      suptitle: str = "") -> Figure:
    """Exemples de simulations : T(t), avec les périodes de chauffe en bandes colorées.

    Args:
        hours: axe du temps (L,).
        t: températures (n, L), en °C.
        u: puissances (n, L), en W.
        titles: un titre par simulation.
    """
    n = len(t)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.9 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, ti, ui, title in zip(axes.flat, t, u, titles):
        on = (ui > 0).numpy()
        ax.fill_between(hours, 0, 1, where=on, transform=ax.get_xaxis_transform(), step="post",
                        color=SERIES[1], alpha=0.18, lw=0, label="chauffage allumé")
        ax.plot(hours, ti.numpy(), color=SERIES[0], label="T intérieure")
        ax.set_title(title, fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel("Temps (h)")
    for row in axes:
        row[0].set_ylabel("T (°C)")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="upper left")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_target_distributions(y: torch.Tensor, targets: list[str]) -> Figure:
    """Histogramme de chaque indicateur, avec sa moyenne et sa médiane."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, (ax, name) in enumerate(zip(axes.flat, targets)):
        v = y[:, j].numpy()
        ax.hist(v, bins=100, color=SERIES[0], alpha=0.85)
        ax.axvline(v.mean(), color=SERIES[1], lw=2, label=f"moyenne {v.mean():.1f}")
        ax.axvline(np.median(v), color=INK, lw=1.5, ls="--", label=f"médiane {np.median(v):.1f}")
        ax.set_title(name)
        ax.set_xlabel(_label(name))
        ax.legend()
    for row in axes:
        row[0].set_ylabel("Nombre de simulations")
    fig.tight_layout()
    return fig


def plot_correlation(y: torch.Tensor, targets: list[str]) -> Figure:
    """Matrice de corrélation des indicateurs, annotée."""
    corr = torch.corrcoef(y.T.double()).numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)), targets, rotation=45, ha="right")
    ax.set_yticks(range(len(targets)), targets)
    for i in range(len(targets)):
        for j in range(len(targets)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.grid(False)
    ax.set_title("Corrélation entre indicateurs")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig


def plot_histograms(values: dict[str, torch.Tensor], xlabels: dict[str, str], bins: int = 60,
                    log: bool = False) -> Figure:
    """Histogrammes d'une série de grandeurs (les NaN sont ignorés)."""
    n = len(values)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3 * nrows), squeeze=False)
    for ax, (name, v) in zip(axes.flat, values.items()):
        v = v[~v.isnan()].numpy()
        ax.hist(v, bins=bins, color=SERIES[0], alpha=0.85)
        ax.set_xlabel(xlabels.get(name, name))
        if log:
            ax.set_yscale("log")
    for row in axes:
        row[0].set_ylabel("Nombre de simulations")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_free_response(hours: np.ndarray, t_free: torch.Tensor) -> Figure:
    """Réponse libre : moyenne et dispersion des simulations sans excitation.

    Args:
        hours: axe du temps (L,).
        t_free: températures des simulations sans excitation (n, L), en °C.
    """
    q = torch.quantile(t_free, torch.tensor([0.05, 0.5, 0.95]), dim=0).numpy()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.fill_between(hours, q[0], q[2], color=SERIES[0], alpha=0.2, lw=0, label="5ᵉ – 95ᵉ centile")
    ax.plot(hours, q[1], color=SERIES[0], label="médiane")
    ax.set(xlabel="Temps (h)", ylabel="T (°C)", title="Réponse libre : simulations sans excitation")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_density(x: torch.Tensor, y: torch.Tensor, xlabel: str, ylabel: str, title: str = "",
                 ax: plt.Axes | None = None) -> Figure:
    """Nuage de points dense, représenté par sa densité (hexagones)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
    else:
        fig = ax.figure
    hb = ax.hexbin(x.numpy(), y.numpy(), gridsize=60, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(hb, ax=ax, label="simulations (échelle log)")
    return fig


def plot_learning_curves(histories: dict[str, list[dict[str, float]]]) -> Figure:
    """Courbes d'apprentissage : validation en trait plein, entraînement en pointillés.

    Les pertes portent sur les cibles standardisées : elles ne se comparent en valeur
    qu'entre expériences de même codage.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for k, (name, history) in enumerate(histories.items()):
        color = SERIES[k % len(SERIES)]
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["val"] for h in history], color=color, label=name)
        ax.plot(epochs, [h["train"] for h in history], color=color, ls=":", lw=1.4)
    ax.set_yscale("log")
    ax.set(xlabel="Époque", ylabel="Perte (cibles standardisées)",
           title="Courbes d'apprentissage — validation (trait plein) et entraînement (pointillés)")
    ax.legend(ncols=2)
    fig.tight_layout()
    return fig


def plot_model_comparison(table: pd.DataFrame, xlabel: str, title: str = "", fmt: str = "{:.2f}") -> Figure:
    """Compare les modèles indicateur par indicateur (un panneau par indicateur).

    Args:
        table: modèles en lignes, indicateurs en colonnes.
        xlabel: nom de la métrique.
        fmt: format des valeurs affichées.
    """
    names = list(table.columns)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 1.6 + 0.62 * len(table)), sharey=True)
    y = np.arange(len(table))[::-1]
    for ax, name in zip(axes.flat, names):
        bars = ax.barh(y, table[name].to_numpy(), height=0.62, color=SERIES[0])
        ax.bar_label(bars, labels=[fmt.format(v) for v in table[name]], padding=3, fontsize=8, color=MUTED)
        ax.set_yticks(y, table.index)
        ax.set_title(_label(name), fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_group_bars(table: pd.DataFrame, ylabel: str, title: str = "") -> Figure:
    """Barres groupées : indicateurs en abscisse, une série par colonne (au plus 4 séries)."""
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(table))
    width = 0.8 / len(table.columns)
    for k, col in enumerate(table.columns):
        bars = ax.bar(x + (k - (len(table.columns) - 1) / 2) * width, table[col], width * 0.92,
                      color=SERIES[k], label=col)
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, color=MUTED, padding=2)
    ax.set_xticks(x, table.index)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel=ylabel, title=title)
    ax.legend(ncols=len(table.columns), loc="lower right")
    fig.tight_layout()
    return fig


def plot_parity(y_true: torch.Tensor, y_pred: torch.Tensor, targets: list[str], title: str = "") -> Figure:
    """Prédiction contre référence pour chaque indicateur (densité), avec la diagonale idéale."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, (ax, name) in enumerate(zip(axes.flat, targets)):
        t, p = y_true[:, j], y_pred[:, j]
        lo, hi = float(t.min()), float(t.max())
        ax.hexbin(t.numpy(), p.numpy(), gridsize=70, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0,
                  extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color=SERIES[1], lw=1.5, label="prédiction parfaite")
        ax.set(xlabel=f"Référence — {_label(name)}", ylabel="Prédiction", title=name,
               xlim=(lo, hi), ylim=(lo, hi))
    axes[0, 0].legend(loc="upper left")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def binned_errors(x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, targets: list[str],
                  n_bins: int = 12) -> pd.DataFrame:
    """RMSE et P95 de l'erreur par tranche d'une grandeur ``x`` (tranches de même effectif).

    Returns:
        Un tableau long : ``centre``, ``n``, ``indicateur``, ``RMSE``, ``P95``.
    """
    keep = ~x.isnan()
    x, err = x[keep], (y_pred - y_true)[keep]
    edges = torch.unique(torch.quantile(x.double(), torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)))
    which = torch.bucketize(x.double(), edges[1:-1])
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() < 20:
            continue
        for j, name in enumerate(targets):
            e = err[m, j].double()
            rows.append({"centre": x[m].double().mean().item(), "n": int(m.sum()), "indicateur": name,
                         "RMSE": e.pow(2).mean().sqrt().item(), "P95": e.abs().quantile(0.95).item()})
    return pd.DataFrame(rows)


def plot_binned_errors(table: pd.DataFrame, targets: list[str], xlabel: str, title: str = "") -> Figure:
    """Erreur en fonction d'une grandeur : RMSE et P95 par tranche, un panneau par indicateur."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex=True)
    for ax, name in zip(axes.flat, targets):
        sub = table[table["indicateur"] == name]
        ax.plot(sub["centre"], sub["RMSE"], marker="o", ms=4, color=SERIES[0], label="RMSE")
        ax.plot(sub["centre"], sub["P95"], marker="o", ms=4, color=SERIES[1], label="P95 de l'erreur absolue")
        ax.set_title(_label(name), fontsize=10)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout()
    return fig
