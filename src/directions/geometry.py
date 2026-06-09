"""
Geometry utilities for comparing directions in activation space.
Covers pairwise cosine similarity, principal angles (subspace overlap),
projection-based monitoring, ablation, and subspace explained-variance.
"""

import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def pairwise_cosine_matrix(
    directions: dict[str, Float[Tensor, " d"]],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute the pairwise cosine similarity matrix for a dict of named direction vectors.
    Returns (matrix, names) where matrix[i,j] = cos_sim(directions[names[i]], directions[names[j]]).
    All inputs are unit-normalised before comparison.
    """
    names = list(directions.keys())
    vecs = t.stack([F.normalize(directions[n].float(), dim=0) for n in names])
    matrix = (vecs @ vecs.T).cpu().numpy()
    return matrix, names


def principal_angle(
    A: Float[Tensor, "n d"],
    B: Float[Tensor, "m d"],
    n_angles: int = 1,
) -> list[float]:
    """
    Compute the first n_angles principal angles (in degrees) between two subspaces A and B.
    A, B are matrices whose rows are basis vectors (need not be orthonormal).
    Uses SVD of A_orth @ B_orth.T.

    For n_angles=1, returns the angle between the nearest directions in the two subspaces.
    0 deg = identical subspaces (along that angle), 90 deg = orthogonal.
    """
    A_orth = t.linalg.qr(A.float().T)[0].T  # (rank_A, d)
    B_orth = t.linalg.qr(B.float().T)[0].T  # (rank_B, d)
    cross = A_orth @ B_orth.T
    S = t.linalg.svdvals(cross)
    S = S.clamp(-1.0, 1.0)[:n_angles]
    return [float(t.acos(s).item() * 180 / np.pi) for s in S]


def project_onto_direction(
    activations: Float[Tensor, "n d"],
    direction: Float[Tensor, " d"],
) -> Float[Tensor, " n"]:
    """Project a batch of activation vectors onto a (unit-normalised) direction."""
    v = F.normalize(direction.float(), dim=0)
    return activations.float() @ v


def ablate_direction(
    activations: Float[Tensor, "... d"],
    direction: Float[Tensor, " d"],
) -> Float[Tensor, "... d"]:
    """
    Project `direction` out of activations (concept ablation / nullspace projection).
    x' = x - (x @ v_hat) * v_hat
    """
    v = F.normalize(direction.float().unsqueeze(0), dim=-1)  # (1, d)
    proj = activations.float() @ v.T  # (..., 1)
    return activations.float() - proj * v


def subspace_explained_variance(
    direction: Float[Tensor, " d"],
    basis: Float[Tensor, "k d"],
) -> float:
    """
    Fraction of a (unit) direction's squared norm explained by the subspace
    spanned by `basis` rows (R^2 of projecting the direction onto the subspace).
    Used for the "is the EM direction inside span{assistant_axis, toxic_persona}?" test.
    """
    v = F.normalize(direction.float(), dim=0)
    Q = t.linalg.qr(basis.float().T)[0]  # (d, k), orthonormal columns
    coeffs = Q.T @ v  # (k,)
    return float((coeffs**2).sum().item())


def plot_cosine_heatmap(matrix: np.ndarray, names: list[str], title: str = ""):
    """Plot a pairwise cosine similarity heatmap using matplotlib. Returns the figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(names)), max(6, len(names))))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(matrix[i, j]) < 0.7 else "white",
            )
    ax.set_title(title or "Pairwise Direction Cosine Similarities")
    plt.tight_layout()
    return fig
