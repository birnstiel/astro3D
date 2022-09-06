import numpy as _np
from scipy.ndimage import laplace as _laplace
import matplotlib.pyplot as _plt
from matplotlib.colors import Normalize as _Normalize
from matplotlib.cm import get_cmap as _get_cmap
from PIL import Image as _Image

from ._lic import lic as _fortran
LIC = _fortran.flic
gen_noise_fast = _fortran.gen_noise_fast

__all__ = ['LIC', 'contrast_enhance']


def contrast_enhance(data, sig=2.0):
    minval, maxval = data.mean() + sig * _np.array([-1, 1]) * data.std()
    minval = max(0.0, minval)
    maxval = min(1.0, maxval)
    output = _np.clip((data - minval) / (maxval - minval), 0.0, 1.0)
    return output


def calc_2D_streamline(p, x, y, vel, data, length=1.0, n_steps=10, direction='forward'):
    """calculate streamline in 2D velocity field

    Parameters
    ----------
    p : array
        initial position as array or list of length 2, (x, y)
    x : array
        regular x grid, shape (nx)
    y : array
        regular y grid, shape (ny)
    vel : array
        velocity in x and y, shape (nx, ny, 2), where one point is (vx, vy)
    data : array
        scalar field of shape (nx, ny), values along the path are returned for LIC
    length : float, optional
        how long the path of the stream line should be, by default 1.0
    n_steps : int, optional
        number of steps to take along the path, by default 10
    direction : str, optional
        direction, by default 'forward', can be
        - 'forward': move in the direction of the velocity
        - 'backward': move in the opposite direction of the velocity
        - 'both': move half a length in both direction


    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    x0, y0 = p
    if direction == 'forward':
        path, ipath, values = _fortran.calc_2d_streamline_forward(x0, y0, x, y, vel, data, length=length, n_steps=n_steps)
    elif direction == 'backward':
        path, ipath, values = _fortran.calc_2d_streamline_forward(x0, y0, x, y, vel, data, length=-length, n_steps=n_steps)
    elif direction == 'both':
        path, ipath, values = _fortran.calc_2d_streamline_bothways(x0, y0, x, y, vel, data, length=length, n_steps=n_steps)
    else:
        raise ValueError("direction needs to be 'forward' or 'both'")
    return path, ipath, values


def LIC_twostage(x, y, vel, generate_plot=False, **kwargs):
    """computes a 2-stage LIC with some contrast enhancement

    Parameters
    ----------
    x : array
        regular x-grid of shape (nx)
    y : array
        regular y-grid of shape (ny)
    vel : array
        2D velocity (vx, vy) of shape (nx, ny, 2)
    generate_plot : bool, optional
        if true, plot the different stages, by default False

    Returns
    -------
    array
        2D LIC pattern
    """

    nx = len(x)
    ny = len(y)
    length = kwargs.pop('length', min(x[-1] - x[0], y[-1] - y[0]) / 5.)
    length = abs(length / 2)

    print(f'length = {length:.2f}')
    noise = gen_noise_fast(nx, ny)
    noise_L = _Normalize()(LIC(noise, x, y, vel, length=length))
    noise_Ll = _laplace(noise_L)
    noise_LlL = _Normalize()(LIC(noise_Ll, x, y, vel, length=length))
    noise_LlLC = contrast_enhance(noise_LlL)

    if generate_plot:
        imgs = [
            'noise',
            'noise_L',
            'noise_Ll',
            'noise_LlL',
            'noise_LlLC',
        ]
        n = len(imgs)
        f, ax = _plt.subplots(n, 2, figsize=(8, 2 * n), dpi=100)

        _loc = locals()

        for i, name in enumerate(imgs):
            cc = ax[i, 0].imshow(_loc[name].T, cmap='gray', norm=_Normalize())
            _plt.colorbar(cc, ax=ax[i, 0])
            ax[i, 0].set_title(name)
            ax[i, 0].set_axis_off()

            ax[i, 1].hist(_loc[name].ravel(), bins=20, fc='k')

    return noise_LlLC


def hsv_mix(scalar, noise, cmap='magma', norm=None):
    """Mixes a color-mapped scalar and a LIC pattern in HSV color space

    Parameters
    ----------
    scalar : array
        2d array of a scalar quantity used for defining the color
    noise : array
        2d array of a LIC pattern, same shape as scalar
    cmap : str, optional
        color map to be used, by default 'magma'
    norm : norm, optional
        norm used to normalize the scalar before color mapping, by default a linear norm is used

    Returns
    -------
    array
        RGB values for the shape of scalar
    """
    if norm is None:
        norm = _Normalize()

    assert scalar.shape == noise.shape, 'scalar and noise need to have the same shape'

    # get the RGB colors for the two images
    img_col = _get_cmap(cmap)(norm(scalar))
    img_lic = _get_cmap('gray')(_Normalize()(noise))

    # get numpy arrays of HSV values
    hsv_col = _np.array(_Image.fromarray(_np.uint8(img_col[:, :, :3] * 255)).convert('HSV'))
    hsv_lic = _np.array(_Image.fromarray(_np.uint8(img_lic[:, :, :3] * 255)).convert('HSV'))

    # put together
    hsv = hsv_col.copy()
    hsv[..., -1] = _np.uint8((_np.float16(hsv_col[..., 2]) + _np.float16(hsv_lic[..., 2])) / 2.0)

    # convert back to RGB-numpy array
    rgb2 = _np.array(_Image.fromarray(hsv, mode='HSV').convert('RGB'))

    return rgb2


def pcolormesh_rgb(x, y, rgb, ax=None, **kwargs):
    """Makes a pcolormesh plot given RGB data

    Parameters
    ----------
    x : array
        x-array
    y : array
        y-array
    rgb : array
        color data, shape (len(x), len(y), 3 or 4)
    ax : axes, optional
        axes into which to plot, by default None

    kwargs : are passed to pcolormesh

    Returns
    -------
    f, ax
        figure and axes object
    """
    if ax is None:
        f, ax = _plt.subplots()
    else:
        f = ax.figure
    col_len = rgb.shape[-1]
    cc = ax.pcolormesh(x, y, rgb[:, :, 0].T, facecolors=rgb.transpose(1, 0, 2).reshape(-1, col_len) / 255, **kwargs)
    cc.set_array(None)
    return f, ax
