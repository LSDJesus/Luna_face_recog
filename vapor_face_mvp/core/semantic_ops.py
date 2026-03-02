"""
Semantic operations on face embeddings.

Utilities to manipulate embeddings along interpretable semantic directions
while preserving identity as measured by cosine similarity.

This complements SemanticProbeTrainer by providing vector-space surgery:
- apply_semantic_shift: add/remove attribute signal along a direction
- project_out_direction: remove attribute component
- safe_alpha_for_cosine: pick alpha to satisfy a cosine floor

All functions assume L2-normalized embeddings and directions.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def ensure_unit(vec: np.ndarray, name: str) -> np.ndarray:
    """Ensure vector is unit length; normalize if needed."""
    n = float(np.linalg.norm(vec))
    if not (0.999 <= n <= 1.001):
        vec = l2_normalize(vec)
    return vec


def apply_semantic_shift(
    embedding: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    mode: str = "add",
) -> np.ndarray:
    """
    Shift an embedding along a semantic direction and re-normalize.

    Args:
        embedding: L2-normalized face embedding (D,)
        direction: L2-normalized semantic direction vector (D,)
        alpha: step size (typ. 0.05 - 0.30)
        mode: 'add' to increase attribute, 'remove' to decrease

    Returns:
        Shifted, L2-normalized embedding (D,)
    """
    e = ensure_unit(embedding, "embedding")
    d = ensure_unit(direction, "direction")
    sign = 1.0 if mode == "add" else -1.0
    e_prime = e + sign * alpha * d
    return l2_normalize(e_prime)


def project_out_direction(embedding: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Remove the component of embedding along 'direction' (orthogonal projection), then renorm.
    """
    e = ensure_unit(embedding, "embedding")
    d = ensure_unit(direction, "direction")
    component = float(np.dot(e, d))
    e_prime = e - component * d
    return l2_normalize(e_prime)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < eps or bn < eps:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def safe_alpha_for_cosine(
    embedding: np.ndarray,
    direction: np.ndarray,
    min_cosine: float = 0.97,
    alpha_max: float = 0.35,
    alpha_step: float = 0.01,
    mode: str = "add",
) -> Tuple[float, float]:
    """
    Find the largest alpha such that cosine(embedding, shifted) >= min_cosine.

    Returns (alpha, resulting_cosine).
    """
    e = ensure_unit(embedding, "embedding")
    d = ensure_unit(direction, "direction")

    best_alpha = 0.0
    best_cos = 1.0
    alpha = 0.0
    while alpha <= alpha_max + 1e-9:
        e_prime = apply_semantic_shift(e, d, alpha, mode=mode)
        cosv = cosine_similarity(e, e_prime)
        if cosv >= min_cosine:
            best_alpha = alpha
            best_cos = cosv
            alpha += alpha_step
        else:
            break
    return best_alpha, best_cos


__all__ = [
    "apply_semantic_shift",
    "project_out_direction",
    "cosine_similarity",
    "safe_alpha_for_cosine",
]
