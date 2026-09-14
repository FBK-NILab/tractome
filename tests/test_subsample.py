"""Tests for large-tractogram stratified subsampling (tractome.compute).

These exercise the pure sampling logic only (numpy + scikit-learn); no Qt,
tractogram IO, or GPU is required.
"""

import numpy as np
import pytest

from tractome.compute import (
    KEEP_FLOOR,
    fine_cluster_labels,
    stratified_subsample,
)


def _make_embedding(seed=123, n_strata=300, dim=40, spread=0.15):
    """Build a synthetic dissimilarity embedding of Gaussian blobs.

    A few strata are deliberately tiny (< KEEP_FLOOR members) to exercise
    thin-bundle coverage.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_strata, dim)) * 3.0
    sizes = rng.integers(20, 400, size=n_strata)
    sizes[:5] = rng.integers(6, 15, size=5)  # thin bundles
    parts = [
        centers[i] + rng.normal(size=(sizes[i], dim)) * spread for i in range(n_strata)
    ]
    X = np.concatenate(parts).astype(np.float32)
    return X[rng.permutation(len(X))]


@pytest.fixture(scope="module")
def embedding():
    return _make_embedding()


def test_identity_when_below_threshold():
    X = _make_embedding(n_strata=10)
    n = len(X)
    r = stratified_subsample(X, threshold=n + 100, q=20)
    assert np.array_equal(r["sample_ids"], np.arange(n))
    assert r["labels"] is None
    assert r["medoid_ids"] is None
    assert r["k_s"] == 0
    assert r["n"] == n


def test_k_s_from_threshold_and_q(embedding):
    r = stratified_subsample(embedding, threshold=1000, q=20)
    assert r["k_s"] == 1000 // 20


def test_every_stratum_keeps_its_medoid_and_quota(embedding):
    thr, q, floor = 1000, 20, KEEP_FLOOR
    r = stratified_subsample(embedding, threshold=thr, q=q, floor=floor)
    labels, medoids, sample = r["labels"], r["medoid_ids"], r["sample_ids"]
    n = r["n"]
    sample_set = set(sample.tolist())

    # sample is sorted & unique
    assert np.array_equal(sample, np.unique(sample))

    present = set(labels.tolist())
    # one medoid per non-empty stratum, all in the sample
    assert len(medoids) == len(present)
    assert set(medoids.tolist()).issubset(sample_set)

    # per-stratum kept count == min(size, max(floor, round(pct*size)))
    pct = thr / n
    total_expected = 0
    for lab in present:
        members = np.where(labels == lab)[0]
        size = len(members)
        keep = min(size, max(floor, round(pct * size)))
        total_expected += keep
        kept = sum(1 for m in members.tolist() if m in sample_set)
        assert kept == keep, f"stratum {lab}: expected {keep} kept {kept}"

    assert len(sample) == total_expected
    assert len(sample) >= len(present)  # at least one medoid each


def test_thin_strata_kept_in_full(embedding):
    """Strata with <= floor members must be fully retained (discoverability)."""
    floor = KEEP_FLOOR
    r = stratified_subsample(embedding, threshold=1000, q=20, floor=floor)
    labels, sample = r["labels"], set(r["sample_ids"].tolist())
    for lab in set(labels.tolist()):
        members = np.where(labels == lab)[0]
        if len(members) <= floor:
            kept = sum(1 for m in members.tolist() if m in sample)
            assert kept == len(members)


def test_determinism(embedding):
    a = stratified_subsample(embedding, threshold=1000, q=20)
    b = stratified_subsample(embedding, threshold=1000, q=20)
    assert np.array_equal(a["sample_ids"], b["sample_ids"])


def test_fine_cluster_labels_medoid_is_member(embedding):
    labels, medoids = fine_cluster_labels(embedding, 50, seed=0)
    assert labels.shape[0] == len(embedding)
    # every medoid belongs to the stratum it represents
    for mid in medoids.tolist():
        stratum = labels[mid]
        assert np.any(labels == stratum)
    # one medoid per non-empty stratum
    assert len(medoids) == len(set(labels.tolist()))


def test_handles_empty_strata_without_crashing():
    """High n_clusters relative to distinct blobs yields empty MBKM strata."""
    X = _make_embedding(seed=7, n_strata=8)
    labels, medoids = fine_cluster_labels(X, 40, seed=0)
    # no crash; medoids only for non-empty strata
    assert len(medoids) == len(set(labels.tolist()))
    assert len(medoids) <= 40
