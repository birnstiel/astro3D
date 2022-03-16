from itertools import repeat
import random
import string
import shutil
from pathlib import Path
import subprocess
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

import volrender

_x0 = [0.1, 0.4, 0.92]
_sigma = [0.03, 0.05, 0.03]
_colors = np.array([
    [1., 0.5, 0., 0.05],
    [0.25, 0.25, 0.75, 0.2],
    [1., 0., 0.25, 0.1]])


def render(data, phi, theta, transferfunction, transparent=False, N=None, bg=0.0):
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

    N : int
        resolution of the interpolated data, by default
        maximum length of the input data dimensions

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
    N = N or n_max
    c = np.linspace(-n_max / 2, n_max / 2, N)
    xo, yo, zo = np.meshgrid(c, c, c)

    # f_interp = RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0.0)

    qxR = xo * np.cos(phi) - yo * np.sin(phi) * np.cos(theta) + zo * np.sin(phi) * np.sin(theta)
    qyR = xo * np.sin(phi) + yo * np.cos(phi) * np.cos(theta) - zo * np.sin(theta) * np.cos(phi)
    qzR = yo * np.sin(theta) + zo * np.cos(theta)
    qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

    # Interpolate onto Camera Grid
    # data_obs = f_interp(qi).reshape((n_max, n_max, n_max))
    data_obs = volrender.fmodule.interpolate(x, y, z, data, qi).reshape((N, N, N))

    res = volrender.fmodule.render(data_obs,
                                   transferfunction.x0,
                                   transferfunction.sigma,
                                   transferfunction.colors,
                                   bg=bg)

    if not transparent:
        res = res[:, :, :3]

    return res


def makeframe(i, data, theta, phi, tf=None, dir='frames', N=None, dpi=300, bg=0.0):
    """Render a frame.

    Parameters
    ----------
    i : int
        integer value of the angle array set in the module.

    data : array  
        3d data set

    theta, phi: float
        the angles

    tf : volrender.TransferFunction
        the transfer function to be used

    N : int
        output resolution, see `render`
    """
    # create the transfer function from the module attributes
    if tf is None:
        tf = volrender.TransferFunction(x0=_x0, sigma=_sigma, colors=_colors)

    # make the plot
    f, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.axis('off')

    image = render(data, phi, theta, tf, N=N, bg=bg)
    ax.imshow(Normalize()(image).data, origin='lower')
    f.savefig(Path(dir) / f'frame_{i:03d}.jpg', bbox_inches='tight', dpi=dpi)
    plt.close(f)


def render_movie(data, theta, phi, ncpu=4, tf=None, fname='movie.mp4', N=None, dpi=200, bg=0.0, cont=False):
    """Renders a movie for the given theta and phi arrays.

    Parameters
    ----------
    data : array
        3d data to be rendered (uniform cartesian)
    theta : array
        the elevation angles for the frames
    phi : array
        the azimuthal angles for all frames, same size as theta
    ncpu : int, optional
        number of CPUs to use, by default 4
    tf : volrender.TransferFunction, optional
        transfer function to be used, by default None
    fname : str, optional
        output file name, by default 'movie.mp4'
    N : int, optional
        resolution of the interpolated data, by default maximum of input dimension
    dpi : int
        resolution of frames in dots per inch
    bg : float
        background color, 1.0 is white, 0.0 is black, default 0.0
    cont : str | bool
        if False, will use temporary directory
        if True will not delete directory,
        if string, continue in the given directory, do not delete
    """
    if cont:
        if cont == True:  # noqa
            cont = 'frames'
        temp_dir = TemporaryDirectory(name=cont, randomize=False, create=False, destroy=False)
    else:
        temp_dir = TemporaryDirectory(name='frames_', create=True, destroy=True)

    if ncpu == 1:
        for i, (_t, _p) in tqdm(enumerate(zip(theta, phi)), total=len(theta)):
            makeframe(i, data, _t, _p, tf, temp_dir.name, dpi=dpi, N=N, bg=bg)

    else:
        with Pool(ncpu) as p:
            # list(tqdm(p.imap(worker, range(n_angles)), total=n_angles))
            list(p.starmap(makeframe, tqdm(zip(
                range(len(theta)),
                repeat(data),
                theta,
                phi,
                repeat(tf),
                repeat(temp_dir.name),
                repeat(N),
                repeat(dpi),
                repeat(bg)), total=len(theta)),
                chunksize=1))

    subprocess.run(('ffmpeg -y -i ' + temp_dir.name + '/frame_%03d.jpg -c:v libx264 -crf 15 -maxrate 6400k -pix_fmt yuv420p -r 24 -bufsize 1835k ' + fname).split())
    if not cont:
        temp_dir.cleanup()


class TemporaryDirectory():
    def __init__(self, name, randomize=True, create=False, destroy=False):
        """Temporary Directory class that needs not be temporary or can already exist.

        Parameters
        ----------
        name : string or path
            path of the folder; can use ~
        randomize : bool, optional
            if true, append random string, by default True
        create : bool, optional
            if True, will create folder if it does not exist, by default False
        destroy : bool, optional
            if true, will destroy folder when exiting context, by default False
        """
        name = (Path() / name).expanduser().name
        if randomize:
            name += '_' + ''.join(random.choices(string.ascii_letters, k=10))
        self._path = Path(name)
        self.create = create
        self.destroy = destroy
        self._check_dir()

    def _check_dir(self):
        if not self._path.is_dir():
            if self.create:
                self._path.mkdir()
            else:
                raise FileNotFoundError('directory not found. use `create` keyword to allow creation.')

    @property
    def name(self):
        return self._path.name

    def __enter__(self):
        self._check_dir()
        return self

    def __exit__(self, type, value, traceback):
        if self.destroy:
            self.cleanup()

    def cleanup(self):
        if self._path.is_dir():
            shutil.rmtree(self._path)
