"""Tests of the compiled kernels in `astro3d.fmodule`.

These define the expected behavior, so they can also serve as reference
if a kernel gets re-implemented (e.g. in numpy/numba).
"""
import numpy as np
import pytest

from astro3d import fmodule


# --- interpolation --------------------------------------------------------

def test_interpolate_linear_exact(rng):
    """trilinear interpolation reproduces a linear function exactly,
    also on non-uniform grids"""
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 2, 10)**2
    z = np.linspace(-1, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vals = 1 + X + 2 * Y + 3 * Z
    pts = rng.uniform([0, 0, -1], [1, 4, 1], (500, 3))
    res = fmodule.interpolate(x, y, z, vals, pts)
    np.testing.assert_allclose(res, 1 + pts @ [1, 2, 3], rtol=1e-12)


def test_interpolate_out_of_bounds():
    x = np.linspace(0, 1, 4)
    vals = np.ones((4, 4, 4))
    pts = np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, -1, 0.5]])
    res = fmodule.interpolate(x, x, x, vals, pts)
    np.testing.assert_allclose(res, [1.0, fmodule.fill_value, fmodule.fill_value])


def test_interpolate_default_threads():
    """must not crash without manually setting threads (bug in the v2 rewrite)"""
    x = np.linspace(0, 1, 4)
    fmodule.interpolate(x, x, x, np.ones((4, 4, 4)), np.full((3, 3), 0.5))
    assert fmodule.numthreads >= 1


# --- dithering ------------------------------------------------------------

@pytest.mark.parametrize('level', [0.1, 0.25, 0.5, 0.9])
def test_dither_preserves_mean(level):
    out = fmodule.dither(np.full((128, 128), level))
    assert set(np.unique(out)) <= {0, 1}
    assert abs(out.mean() - level) < 0.01


def test_dither_colors(rng):
    levels = np.array([0.1, 0.2, 0.3])
    out = fmodule.dither_colors(np.ones((128, 128, 3)) * levels)
    assert set(np.unique(out)) <= {0, 1}
    # every pixel gets at most one material
    assert out.sum(-1).max() <= 1
    np.testing.assert_allclose(out.mean((0, 1)), levels, atol=0.01)


# --- view rendering -------------------------------------------------------

@pytest.mark.parametrize('nthreads', [1, 2, 4])
def test_compute_view_analytic(nthreads, numthreads):
    """a uniform block of n voxels blends as color * (1 - tf**n) over black"""
    fmodule.numthreads = nthreads
    nz, n_tauone = 10, 5
    data = np.zeros((16, 16, nz, 3), dtype=np.int32, order='F')
    data[..., 0] = 200
    empty = np.array([[0, 0, 0]], dtype=np.int32)
    image = fmodule.compute_view(data, 1, nz, 1, n_tauone, empty, np.zeros(3))
    expected = 200 * (1 - np.exp(-nz / n_tauone))
    np.testing.assert_allclose(image[..., 0], expected)
    np.testing.assert_allclose(image[..., 1:], 0)


def test_compute_view_skips_empty_colors():
    """voxels with an `empty` color are transparent: background stays"""
    data = np.full((4, 4, 3, 3), 255, dtype=np.int32, order='F')
    empty = np.array([[255, 255, 255]], dtype=np.int32)
    bg = np.array([10.0, 20.0, 30.0])
    image = fmodule.compute_view(data, 1, 3, 1, 5, empty, bg)
    np.testing.assert_allclose(image, np.broadcast_to(bg, image.shape))


# --- color detection ------------------------------------------------------

def test_get_colors(rng):
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])
    stack = palette[rng.integers(0, 3, (20, 10, 5))]
    nresults, output = fmodule.get_colors(stack, 8)
    assert nresults == 3
    found = {tuple(c) for c in output[:nresults]}
    assert found == {tuple(c) for c in palette}


# --- streamlines ----------------------------------------------------------

def test_mark_streamline():
    """a straight line along x through the center marks a band of width 2r"""
    x = np.linspace(0, 1, 51)
    r = 0.1
    line = np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    mask = fmodule.mark_streamline(x, x, 0.5, r, line)
    X, Y = np.meshgrid(x, x, indexing='ij')
    inside = (X >= 0.2) & (X <= 0.8) & (np.abs(Y - 0.5) < r - 1e-9)
    outside = np.hypot(np.maximum(np.abs(X - 0.5) - 0.3, 0), Y - 0.5) > r + 1e-9
    assert mask[inside].all()
    assert not mask[outside].any()


# --- point clouds ---------------------------------------------------------

@pytest.mark.parametrize('x0, dx, expected', [
    (5.0, 2.0, (4, 8)),     # grid points 3...7 (0-based) -> 4...8 (1-based)
    (5.0, 1.5, (5, 7)),     # 3.5...6.5 -> points 4, 5, 6
    (0.0, 3.0, (1, 4)),     # clipped at the lower grid end
    (9.0, 5.0, (5, 11)),    # clipped at the upper grid end
    (10.0, 0.0, (11, 11)),  # exactly on the last point
    (5.5, 0.2, None),       # no grid point within: empty range
])
def test_cutoff_range(x0, dx, expected):
    """indices (1-based, inclusive) of grid points within x0 +- dx"""
    xg = np.arange(11.0)
    # inout arguments are modified in place, so they need to be arrays
    i0, i1 = np.array(1, dtype=np.int32), np.array(1, dtype=np.int32)
    fmodule.cutoff_range(xg, x0, dx, i0, i1)
    i0, i1 = int(i0), int(i1)
    if expected is None:
        assert i0 > i1
    else:
        assert (i0, i1) == expected

def test_point_cloud_single_point():
    g = np.linspace(-1, 1, 21)
    image = fmodule.point_cloud(g, g, g, [0.0], [0.0], [0.0], [0.2], 5)
    image = image[..., 0]
    assert np.unravel_index(image.argmax(), image.shape) == (10, 10, 10)
    # same profile along all axes
    np.testing.assert_allclose(image[:, 10, 10], image[10, :, 10])
    np.testing.assert_allclose(image[:, 10, 10], image[10, 10, :])


def test_point_cloud_symmetric_cutoff():
    g = np.linspace(-1, 1, 21)
    image = fmodule.point_cloud(g, g, g, [0.0], [0.0], [0.0], [0.1], 3)[..., 0]
    np.testing.assert_allclose(image, image[::-1, :, :])


@pytest.mark.parametrize('nthreads', [2, 8])
def test_point_cloud_colored_slice_threads(rng, nthreads, numthreads):
    """result must not depend on the number of threads (race condition fixed)"""
    n = 5000
    xi, yi, zi = rng.uniform(0, 1, (3, n))
    sigma = np.full(n, 0.03)
    weights = rng.uniform(0, 1, (n, 4))
    g = np.linspace(0, 1, 48)

    fmodule.numthreads = 1
    slice1, alpha1, mean1 = fmodule.point_cloud_colored_slice(
        g, g, 0.5, xi, yi, zi, sigma, 5, weights=weights, ncol=3)
    fmodule.numthreads = nthreads
    slice2, alpha2, mean2 = fmodule.point_cloud_colored_slice(
        g, g, 0.5, xi, yi, zi, sigma, 5, weights=weights, ncol=3)

    # floats: only round-off from a different summation order
    np.testing.assert_allclose(alpha1, alpha2, rtol=1e-10)
    np.testing.assert_allclose(mean1, mean2, rtol=1e-10)
    # colors are truncated to integers 0...255, so round-off can flip by 1
    assert np.abs(slice1.astype(int) - slice2).max() <= 1
