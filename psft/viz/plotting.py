"""Headless plotting helpers (matplotlib Agg).

Each helper returns the path of the file it saved.  These are useful for the
example scripts and tests, where we want reproducible visual evidence without
launching a GUI.
"""
from __future__ import annotations
from typing import Optional, Tuple
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scalar_slice(field: np.ndarray, axis: str, index: int, path: str,
                      title: str = "", cmap: str = "viridis") -> str:
    if axis == "x":
        slc = field[index, :, :]
    elif axis == "y":
        slc = field[:, index, :]
    else:
        slc = field[:, :, index]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(slc.T, origin="lower", cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title or f"slice {axis}={index}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_vector_field_2d(Vx: np.ndarray, Vy: np.ndarray, path: str,
                         title: str = "", step: int = 4) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    Y, X = np.mgrid[0:Vx.shape[0], 0:Vx.shape[1]]
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              Vx[::step, ::step], Vy[::step, ::step])
    ax.set_aspect("equal")
    ax.set_title(title or "vector field")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_phase_winding(field_complex: np.ndarray, slice_z: int, path: str,
                       title: str = "") -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    phase = np.angle(field_complex[:, :, slice_z])
    im = ax.imshow(phase.T, origin="lower", cmap="hsv", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im, ax=ax, label="phase")
    ax.set_title(title or "phase")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_radial_profile(rs: np.ndarray, ys: np.ndarray, path: str,
                        title: str = "", xlabel: str = "r", ylabel: str = "f(r)",
                        log_x: bool = False, log_y: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rs, ys, lw=2)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_animation_frames(fields: list, dir_path: str, basename: str,
                          plot_kwargs: Optional[dict] = None) -> list:
    plot_kwargs = plot_kwargs or {}
    paths = []
    for i, field in enumerate(fields):
        p = os.path.join(dir_path, f"{basename}_{i:04d}.png")
        plot_scalar_slice(field, axis="z", index=field.shape[-1] // 2,
                          path=p, **plot_kwargs)
        paths.append(p)
    return paths
