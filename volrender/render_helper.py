import argparse
from itertools import repeat
import tempfile
from pathlib import Path
import subprocess
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
# from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import volrender

A = [0.15, 0.2, 0.35]
x0 = [0.1, 0.4, 0.92]
sigma = [0.03, 0.05, 0.03]
colors = np.array([
    [1., 0.5, 0., 0.05],
    [0.25, 0.25, 0.75, 0.2],
    [1., 0., 0.25, 0.1]])


def render(data, phi, theta, transferfunction):
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

    return volrender.fmodule.render(data_obs,
                                    transferfunction.x0,
                                    transferfunction.A,
                                    transferfunction.sigma,
                                    transferfunction.colors)


def makeframe(i, data, theta, phi, tf=None, dir='frames'):
    """Render a frame.

    Parameters
    ----------
    i : int
        integer value of the angle array set in the module.
    """
    # create the transfer function from the module attributes
    if tf is None:
        tf = volrender.TransferFunction(x0=x0, A=A, sigma=sigma, colors=colors)

    # make the plot
    f, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.axis('off')

    image = render(data, phi, theta, tf)
    ax.imshow(Normalize()(image).data)
    f.savefig(Path(dir) / f'frame_{i:03d}.jpg', bbox_inches='tight', dpi=200)
    plt.close(f)


def main():
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description='simple volume rendering example', formatter_class=RTHF)
    PARSER.add_argument('-f', '--field', help='if dictionary based npz file is used, use this field from the file', type=str, default=None)
    PARSER.add_argument('filename', help='which .npz file to read', type=str)
    PARSER.add_argument('-o', '--output', help='how to name output file', type=str, default='movie.mp4')
    PARSER.add_argument('-t0', '--theta0', help='initial value of theta', type=float, default=45.)
    PARSER.add_argument('-t1', '--theta1', help='final value of theta', type=float, default=45.)
    PARSER.add_argument('-p0', '--phi0', help='initial value of phi', type=float, default=0.)
    PARSER.add_argument('-p1', '--phi1', help='final value of phi', type=float, default=180.)
    PARSER.add_argument('-N', '--N', help='number of frames', type=int, default=180)
    PARSER.add_argument('-c', '--cpu', help='number of cpus', type=int, default=4)
    ARGS = PARSER.parse_args()

    print('reading data ... ', end='')
    if ARGS.field is None:
        data = np.load(ARGS.filename)
    else:
        with np.load(ARGS.filename) as f:
            data = f[ARGS.field]
    print('Done!')

    vmax = data.max()
    datacube = LogNorm(vmin=vmax * 1e-4, vmax=vmax, clip=True)(data.ravel()).reshape(data.shape).data

    theta = np.linspace(ARGS.theta0, ARGS.theta1, ARGS.N)
    phi = np.linspace(ARGS.phi0, ARGS.phi1, ARGS.N)

    render_movie(datacube, theta, phi, ncpu=ARGS.cpu, fname=ARGS.output)


def render_movie(data, theta, phi, ncpu=4, tf=None, fname='movie.mp4'):

    temp_dir = tempfile.TemporaryDirectory(dir='.', prefix='frames_')

    if ncpu == 1:
        for i, (_t, _p) in tqdm(enumerate(zip(theta, phi)), total=len(theta)):
            makeframe(i, data, _t, _p, tf, temp_dir.name)

    else:
        with Pool(ncpu) as p:
            # list(tqdm(p.imap(worker, range(n_angles)), total=n_angles))
            list(p.starmap(makeframe, zip(
                range(len(theta)),
                repeat(data),
                theta,
                phi,
                repeat(tf),
                repeat(temp_dir.name)
            )))

    subprocess.run(('ffmpeg -y -i ' + temp_dir.name + '/frame_%03d.jpg -c:v libx264 -crf 15 -maxrate 400k -pix_fmt yuv420p -r 20 -bufsize 1835k ' + fname).split())
    temp_dir.cleanup()


if __name__ == '__main__':
    main()
