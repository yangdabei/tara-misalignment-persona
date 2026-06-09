"""Unit tests for src/directions/geometry.py and mean-diff normalisation."""

import sys
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.directions.geometry import (
    ablate_direction,
    pairwise_cosine_matrix,
    principal_angle,
    project_onto_direction,
    subspace_explained_variance,
)


def test_pairwise_cosine_identity():
    """cosine(v, v) == 1.0 for unit vectors (diagonal of the matrix)."""
    t.manual_seed(0)
    directions = {f"v{i}": F.normalize(t.randn(64), dim=0) for i in range(4)}
    matrix, names = pairwise_cosine_matrix(directions)
    assert names == list(directions.keys())
    assert t.allclose(t.tensor(matrix).diagonal(), t.ones(4), atol=1e-5)
    # Symmetry
    assert t.allclose(t.tensor(matrix), t.tensor(matrix).T, atol=1e-6)


def test_pairwise_cosine_orthogonal():
    """Orthogonal basis vectors have off-diagonal cosine 0."""
    directions = {"e0": t.eye(8)[0], "e1": t.eye(8)[1]}
    matrix, _ = pairwise_cosine_matrix(directions)
    assert abs(matrix[0, 1]) < 1e-6


def test_ablate_direction_orthogonal():
    """After ablation, dot(result, v) ~ 0."""
    t.manual_seed(1)
    activations = t.randn(10, 64)
    v = t.randn(64)
    ablated = ablate_direction(activations, v)
    projections = project_onto_direction(ablated, v)
    assert t.allclose(projections, t.zeros(10), atol=1e-4)


def test_ablate_preserves_orthogonal_component():
    """Ablation must not change components orthogonal to v."""
    v = t.eye(4)[0]
    x = t.tensor([[3.0, 2.0, 1.0, 0.0]])
    ablated = ablate_direction(x, v)
    assert t.allclose(ablated, t.tensor([[0.0, 2.0, 1.0, 0.0]]), atol=1e-6)


def test_mean_diff_normalization():
    """A mean-diff style direction is unit-norm after F.normalize."""
    t.manual_seed(2)
    h_aligned = t.randn(20, 64)
    h_misaligned = t.randn(20, 64) + 0.5
    direction = F.normalize(h_misaligned.mean(0) - h_aligned.mean(0), dim=0)
    assert t.allclose(direction.norm(), t.tensor(1.0), atol=1e-5)


def test_principal_angle_identical_and_orthogonal():
    """Identical 1D subspaces -> 0 deg; orthogonal -> 90 deg."""
    v = t.eye(16)[0].unsqueeze(0)
    w = t.eye(16)[1].unsqueeze(0)
    assert principal_angle(v, v.clone())[0] == pytest.approx(0.0, abs=1e-3)
    assert principal_angle(v, w)[0] == pytest.approx(90.0, abs=1e-3)
    # Sign-invariance: a subspace equals its negation
    assert principal_angle(v, -v)[0] == pytest.approx(0.0, abs=1e-3)


def test_subspace_explained_variance():
    """A vector inside the subspace has R^2 = 1; orthogonal has R^2 = 0."""
    basis = t.eye(8)[:2]  # span{e0, e1}
    inside = F.normalize(t.tensor([1.0, 2.0, 0, 0, 0, 0, 0, 0]), dim=0)
    outside = t.eye(8)[5]
    assert subspace_explained_variance(inside, basis) == pytest.approx(1.0, abs=1e-5)
    assert subspace_explained_variance(outside, basis) == pytest.approx(0.0, abs=1e-6)
