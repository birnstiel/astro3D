from multiprocessing import Pool
import argparse
from pathlib import Path

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from volrender import Renderer
from .image_stack import process
from .render_helper import render_movie


def volrender_CLI():
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description='simple volume rendering example', formatter_class=RTHF)
    PARSER.add_argument('-f', '--field', help='if dictionary based npz file is used, use this field from the file', type=str, default='arr_0')
    PARSER.add_argument('-d', '--diagnostics', help='show diagnostics in interactive view', action='store_true', default=False)
    PARSER.add_argument('filename', help='which .npz file to read', type=str)
    ARGS = PARSER.parse_args()

    print('reading data ... ', end='')
    if Path(ARGS.filename).suffix.lower() == '.npy':
        data = np.load(ARGS.filename)
    else:
        with np.load(ARGS.filename) as f:
            data = f[ARGS.field]
    print('Done!')

    vmax = data.max()
    datacube = LogNorm(vmin=vmax * 1e-4, vmax=vmax, clip=True)(data.ravel()).reshape(data.shape).data

    Renderer(datacube, interactive=True, diagnostics=ARGS.diagnostics)
    plt.show()


def image_stack_CLI():
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description='simple volume rendering example', formatter_class=RTHF)
    PARSER.add_argument('-f', '--field', help='if dictionary based npz file is used, use this field from the file', type=str, default='arr_0')
    PARSER.add_argument('filename', help='which .npz file to read', type=str)
    PARSER.add_argument('-L', '--height', help='height in cm', type=float, default=10.)
    PARSER.add_argument('-x', '--dpi_x', help='dpi in x', type=float, default=600.)
    PARSER.add_argument('-y', '--dpi_y', help='dpi in y', type=float, default=600.)
    PARSER.add_argument('-z', '--dpi_z', help='dpi in z', type=float, default=1200)
    PARSER.add_argument('-c', '--cpus', help='how many cores to use', type=int, default=1)
    PARSER.add_argument('-o', '--output', help='output folder', type=str, default='slices')
    ARGS = PARSER.parse_args()

    print('reading data ... ', end='')
    if Path(ARGS.filename).suffix.lower() == '.npy':
        data = np.load(ARGS.filename)
    else:
        with np.load(ARGS.filename) as f:
            data = f[ARGS.field][()]
    print('Done!')

    if ARGS.cpus == 1:
        pool = None
    else:
        pool = Pool(processes=ARGS.cpus)

    # vmax = data.max()
    # datacube = LogNorm(vmin=vmax * 1e-4, vmax=vmax, clip=True)(data.ravel()).reshape(data.shape).data

    process(data, height=ARGS.height,
            dpi_x=ARGS.dpi_x, dpi_y=600, dpi_z=1200,
            output_dir=ARGS.output, norm=None, pool=pool)


def render_movie_CLI():
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description='simple volume rendering example', formatter_class=RTHF)
    PARSER.add_argument('-f', '--field', help='if dictionary based npz file is used, use this field from the file', type=str, default='arr_0')
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
    if Path(ARGS.filename).suffix.lower() == '.npy':
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
