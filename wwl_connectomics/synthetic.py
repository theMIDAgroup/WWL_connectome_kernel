"""Synthetic FC/DTI generators used for demos, smoke tests and the benchmark scenarios."""

import numpy as np


def make_group_A(seed, n, noise=0.15):
    """Strong DMN group: strong intra-block FC (first n//2 regions), dense DTI."""
    rng = np.random.RandomState(seed)
    mid = n // 2

    b1    = rng.uniform(0.70, 0.95, (mid, mid));     b1 = (b1 + b1.T) / 2
    b2    = rng.uniform(0.25, 0.55, (n - mid, n - mid)); b2 = (b2 + b2.T) / 2
    cross = rng.uniform(0.05, 0.20, (mid, n - mid))
    FC    = np.block([[b1, cross], [cross.T, b2]])
    FC   += rng.normal(0, noise, (n, n))
    FC    = np.clip((FC + FC.T) / 2, 0, 1)
    np.fill_diagonal(FC, 1.0)

    dti = rng.uniform(0.4, 1.0, (n, n))
    dti[:mid, :mid] *= 1.3
    dti = np.clip(dti, 0, 1)
    dti[dti < 0.45] = 0
    dti = (dti + dti.T) / 2
    np.fill_diagonal(dti, 0)
    DTI = dti / (dti.max() + 1e-9)
    return FC, DTI


def make_group_B(seed, n, noise=0.15):
    """Flat/distributed group: no dominant FC block, sparser DTI."""
    rng = np.random.RandomState(seed + 100)
    mid = n // 2

    b1    = rng.uniform(0.35, 0.65, (mid, mid));     b1 = (b1 + b1.T) / 2
    b2    = rng.uniform(0.35, 0.65, (n - mid, n - mid)); b2 = (b2 + b2.T) / 2
    cross = rng.uniform(0.30, 0.55, (mid, n - mid))
    FC    = np.block([[b1, cross], [cross.T, b2]])
    FC   += rng.normal(0, noise, (n, n))
    FC    = np.clip((FC + FC.T) / 2, 0, 1)
    np.fill_diagonal(FC, 1.0)

    dti = rng.uniform(0, 1, (n, n))
    dti[dti < 0.65] = 0
    dti = (dti + dti.T) / 2
    np.fill_diagonal(dti, 0)
    DTI = dti / (dti.max() + 1e-9)
    return FC, DTI
