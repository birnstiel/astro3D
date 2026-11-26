"""End-to-end tests: data cube -> PNG slices -> image stack.

Tests marked `xfail(strict=True)` document known bugs. Once a bug is fixed,
the test unexpectedly passes, pytest reports a failure, and the marker
should be removed.
"""
import numpy as np
import pytest
from matplotlib.colors import Normalize
from PIL import Image

from astro3d import fmodule, image_stack

# print size: height=0.03 cm at 941 dpi gives 11 layers
SIZE = dict(height=0.03, dpi_x=600, dpi_y=300, dpi_z=941)


@pytest.fixture
def slices(small_cube, tmp_path):
    out = tmp_path / 'slices'
    norm = Normalize(vmin=0, vmax=small_cube.max())
    image_stack.process(small_cube, output_dir=out, norm=norm, **SIZE)
    return out


def test_process_writes_slices(slices):
    files = sorted(slices.glob('slice_*.png'))
    assert len(files) == int(SIZE['dpi_z'] * SIZE['height'] / 2.54)


def test_image_stack_from_slices(slices):
    stack = image_stack.IStack(slices)
    n_files = len(list(slices.glob('slice_*.png')))
    assert stack.imgs.ndim == 4
    assert stack.imgs.shape[2] == n_files
    assert stack.imgs.dtype == np.uint8
    colors = image_stack.check_colors(stack.imgs, stride=1)
    assert 1 <= len(colors) <= 8


def _read_slices(path):
    return {f.name: np.array(Image.open(f)) for f in sorted(path.glob('slice_*.png'))}


@pytest.mark.parametrize('norm', ['log', 'lin', None])
def test_process_norm_string(small_cube, tmp_path, norm):
    image_stack.process(small_cube, output_dir=tmp_path / 's', norm=norm, **SIZE)
    assert len(list((tmp_path / 's').glob('slice_*.png'))) > 0


def test_process_does_not_modify_norm(small_cube, tmp_path):
    norm = Normalize()
    image_stack.process(small_cube, output_dir=tmp_path / 's', norm=norm, vmin=0, vmax=1, **SIZE)
    assert norm.vmin is None and norm.vmax is None


def test_process_with_grid(small_cube, slices, tmp_path):
    """physical grid coordinates give the same slices as the default index grid"""
    x = np.linspace(-3, 5, small_cube.shape[0])
    norm = Normalize(vmin=0, vmax=small_cube.max())
    image_stack.process(small_cube, output_dir=tmp_path / 's', norm=norm, x=x, y=x, z=x, **SIZE)
    ref = _read_slices(slices)
    res = _read_slices(tmp_path / 's')
    assert ref.keys() == res.keys()
    for name in ref:
        np.testing.assert_array_equal(ref[name], res[name])


def test_process_single_layer(small_cube, slices, tmp_path):
    """a selected layer is named by its index in the full stack"""
    norm = Normalize(vmin=0, vmax=small_cube.max())
    image_stack.process(small_cube, output_dir=tmp_path / 's', norm=norm, iz=[5], **SIZE)
    res = _read_slices(tmp_path / 's')
    assert list(res) == ['slice_0005.png']
    np.testing.assert_array_equal(res['slice_0005.png'], _read_slices(slices)['slice_0005.png'])


def test_process_threads_match_serial(small_cube, tmp_path, numthreads):
    """slices computed in parallel threads are identical to serial ones,
    and the fortran thread count is restored afterwards"""
    norm = Normalize(vmin=0, vmax=small_cube.max())
    fmodule.numthreads = 3
    for n in [1, 4]:
        image_stack.process(small_cube, output_dir=tmp_path / f't{n}', norm=norm,
                            n_threads=n, **SIZE)
        assert int(fmodule.numthreads) == 3
    ref = _read_slices(tmp_path / 't1')
    res = _read_slices(tmp_path / 't4')
    assert ref.keys() == res.keys() and len(ref) > 1
    for name in ref:
        np.testing.assert_array_equal(ref[name], res[name])


@pytest.mark.xfail(raises=AttributeError, strict=True,
                   reason='known bug: calls removed fmodule.point_cloud_colored')
def test_point_cloud_alpha(rng):
    xi, yi, zi = rng.uniform(0, 1, (3, 50))
    image_stack.image_stack_from_point_cloud(
        xi, yi, zi, n=8, weights=rng.uniform(0, 1, (50, 4)), alpha_method=True)


@pytest.mark.xfail(raises=TypeError, strict=True,
                   reason='known bug: passes unknown `nweights` to fmodule.point_cloud')
def test_point_cloud(rng):
    xi, yi, zi = rng.uniform(0, 1, (3, 50))
    image_stack.image_stack_from_point_cloud(xi, yi, zi, n=8)
