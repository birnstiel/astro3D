from itertools import repeat
import tempfile
from pathlib import Path
import subprocess
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import volrender

_x0 = [0.1, 0.4, 0.92]
_sigma = [0.03, 0.05, 0.03]
_colors = np.array([
    [1., 0.5, 0., 0.05],
    [0.25, 0.25, 0.75, 0.2],
    [1., 0., 0.25, 0.1]])


def render(data, phi, theta, transferfunction, transparent=False):
    """render the data from azimuth `phi`, elevation `theta` using the given transfer function.

    Parameters
    ----------
    data : array
        the data array, ndim=3
    phi : float
        azimuthal angle in degree
    theta : float
        polar angle in degree
    transferfunction : `volrender.volrender.Transferfunction`
        the transfer function returning a RGBA value for each input.

    Returns
    -------
    RGBA image
        NxNx4 array to be plotted as image
    """
    phi, theta = np.deg2rad([phi, theta])

    nx, ny, nz = data.shape
    x = np.linspace(-nx / 2, nx / 2, nx)
    y = np.linspace(-ny / 2, ny / 2, ny)
    z = np.linspace(-nz / 2, nz / 2, nz)

    n_max = max(data.shape)
    c = np.linspace(-n_max / 2, n_max / 2, n_max)
    xo, yo, zo = np.meshgrid(c, c, c)

    # f_interp = RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0.0)

    qxR = xo * np.cos(phi) - yo * np.sin(phi) * np.cos(theta) + zo * np.sin(phi) * np.sin(theta)
    qyR = xo * np.sin(phi) + yo * np.cos(phi) * np.cos(theta) - zo * np.sin(theta) * np.cos(phi)
    qzR = yo * np.sin(theta) + zo * np.cos(theta)
    qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

    # Interpolate onto Camera Grid
    # data_obs = f_interp(qi).reshape((n_max, n_max, n_max))
    data_obs = volrender.fmodule.interpolate(x, y, z, data, qi).reshape((n_max, n_max, n_max))

    res = volrender.fmodule.render(data_obs,
                                   transferfunction.x0,
                                   transferfunction.sigma,
                                   transferfunction.colors)

    if not transparent:
        res = res[:, :, :3]

    return res


def makeframe(i, data, theta, phi, tf=None, dir='frames'):
    """Render a frame.

    Parameters
    ----------
    i : int
        integer value of the angle array set in the module.
    """
    # create the transfer function from the module attributes
    if tf is None:
        tf = volrender.TransferFunction(x0=_x0, sigma=_sigma, colors=_colors)

    # make the plot
    f, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.axis('off')

    image = render(data, phi, theta, tf)
    ax.imshow(Normalize()(image).data, origin='lower')
    f.savefig(Path(dir) / f'frame_{i:03d}.jpg', bbox_inches='tight', dpi=200)
    plt.close(f)


def render_movie(data, theta, phi, ncpu=4, tf=None, fname='movie.mp4'):

    temp_dir = tempfile.TemporaryDirectory(dir='.', prefix='frames_')

    if ncpu == 1:
        for i, (_t, _p) in tqdm(enumerate(zip(theta, phi)), total=len(theta)):
            makeframe(i, data, _t, _p, tf, temp_dir.name)

    else:
        with Pool(ncpu) as p:
            # list(tqdm(p.imap(worker, range(n_angles)), total=n_angles))
            list(p.starmap(makeframe, tqdm(zip(
                range(len(theta)),
                repeat(data),
                theta,
                phi,
                repeat(tf),
                repeat(temp_dir.name)), total=len(theta)),
                chunksize=ncpu))

    subprocess.run(('ffmpeg -y -i ' + temp_dir.name + '/frame_%03d.jpg -c:v libx264 -crf 15 -maxrate 400k -pix_fmt yuv420p -r 20 -bufsize 1835k ' + fname).split())
    temp_dir.cleanup()
