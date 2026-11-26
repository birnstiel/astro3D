import numpy as np
import pytest

from astro3d import fmodule


@pytest.fixture
def rng():
    """reproducible random numbers"""
    return np.random.default_rng(42)


@pytest.fixture
def numthreads():
    """allows a test to change `fmodule.numthreads`; restores it afterwards"""
    old = int(fmodule.numthreads)
    yield
    fmodule.numthreads = old


@pytest.fixture
def small_cube(rng):
    """a small, smooth, positive 3D data set (a few gaussian blobs)"""
    n = 16
    x = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    data = np.full((n, n, n), 1e-2)
    for cx, cy, cz in rng.uniform(0.2, 0.8, (4, 3)):
        data += np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * 0.1**2))
    return data
