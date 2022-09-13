from pathlib import Path
from itertools import repeat
import imageio
from multiprocessing import Pool, cpu_count

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

from .. import fmodule


def makeslice(iz, z2, f_interp, coords, norm, path, bits=32, fg=None, bg=None):
    """
    Prints out one slice of the image (index `iz` within grid `z2`) and stores it as
    a file `slice_{iz:04d}.png` in the folder `path`. Black and white in the image can be
    replace by `fg` and `bg` which are RGBA colors such as `[255, 255, 255, 255]`.


    iz : int
        slice index within `z2`

    z2 : array
        the new vertical coordinate array

    f_inter : callable
        the interpolation function of (x,y,z)

    norm : callable
        the normalization function that maps density to 0...1

    bits : int
        bit depth for output image. 1 should be enough, but 32 seems to be needed for fit.technology

    path : str | Path
        the path into which to store the images

    bg : array
        background color

    fg : array
        foreground color

    """
    # update coordinates - only last entry changes
    _x, _y, _z = coords
    _z = np.array([[[z2[iz]]]])

    # interpolate: note that we transpose as this is how the image will be safed
    new_layer = f_interp((_x, _y, _z))[:, :, 0].T

    # normalize, convert to grayscale image
    layer_norm = np.array(norm(new_layer))
    layer_dither = fmodule.dither(layer_norm)

    # set the foreground and background
    fg = fg or [0, 0, 0, 255]
    bg = bg or [255, 255, 255, 255]

    # replace colors
    im = np.where((layer_dither == 0.0)[:, :, None], bg, fg)

    # save as png
    imageio.imwrite(path / f'slice_{iz:04d}.png', np.uint8(im))


def process(data, height=10, dpi_x=600, dpi_y=600, dpi_z=1200, output_dir='slices',
            norm=None, pool=None, vmin=None, vmax=None, iz=None, fg=None, bg=None, bits=32):
    """produce the image stack for 3D printing the given dataset

    Parameters
    ----------
    data : numpy array
        3D data set
    height : int, optional
        height of the cube in cm, by default 10
    dpi_x : int, optional
        x resolution of the printer, by default 600
    dpi_y : int, optional
        y resolution of the printer, by default 600
    dpi_z : int, optional
        z resolution of the printer, by default 1200
    output_dir : str, optional
        path of output directory, by default 'slices'
    norm : norm or None or str, optional
        use the given norm,
        or a lognorm if set to 'log' or None
        or linear norm if set to 'lin', by default None
    pool : None or pool object, optional
        pool for parallel processing, by default None
    vmin : None or float, optional
        minimum value if norm is not given, by default None
    vmax : None or float, optional
        maximum value if norm is not given, by default None
    iz : None or int or int-array, optional
        if int/int array is given, only this/these slice index/indices are produced, by default None

    fg, bg, bits : see `make_slice`

    Raises
    ------
    ValueError
        if norm is invalid string
    """

    if norm is None or isinstance(norm, str):

        if isinstance(norm, str) and norm in ['lin', 'linear']:
            print('using linear norm ', end='')
            Norm = Normalize
        elif isinstance(norm, str) and norm in ['log', 'logarithmic']:
            print('using logarithmic norm ', end='')
            Norm = LogNorm
        elif norm is None:
            Norm = LogNorm
            print('no norm given, using logarithmic norm ', end='')
        else:
            raise ValueError('norm is not a valid input argument for normalization')

        vmax = vmax or 10**np.ceil(np.log10(data.max()))
        vmin = vmin or 1e-2 * vmax
        print('from {vmin:.2g} to {vmax:.2g}')
        norm = Norm(vmin=vmin, vmax=vmax, clip=True)

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])

    f_interp = RegularGridInterpolator((x, y, z), data)

    # calculate new grids

    n_z = int(dpi_z * height / 2.54)
    n_x = int(n_z * len(x) / len(z) / dpi_z * dpi_x)
    n_y = int(n_z * len(y) / len(z) / dpi_z * dpi_y)

    n_x += n_x % 2  # add 1 to make it even if it isn't
    n_y += n_y % 2  # add 1 to make it even if it isn't

    x2 = np.linspace(0, data.shape[0] - 1, n_x)
    y2 = np.linspace(0, data.shape[1] - 1, n_y)
    z2 = np.linspace(0, data.shape[2] - 1, n_z)
    _x, _y, _z = np.meshgrid(x2, y2, z2[0], sparse=True, indexing='ij')
    coords = (_x, _y, _z)

    print(f'  original data: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}')
    print(f'interpoation to: {n_x} x {n_y} x {n_z}')
    print(f'print size: {n_x * 2.54 / dpi_x:.2f} x {n_y * 2.54 / dpi_y:.2f} x {n_z *2.54 / dpi_z:.2f} cm')
    print(f'saving into {output_dir}')
    path = Path(output_dir)

    if not path.is_dir():
        path.mkdir()
    else:
        files = list(path.glob('slice*.png'))
        if len(files) > 0:
            print('directory exists, deleting old files')
            for file in files:
                file.unlink()

    if iz is not None:
        z2 = z2[np.array(iz, ndmin=1)]

    n = len(z2)

    if pool is None:
        list(tqdm(
            map(makeslice, range(n),
                repeat(z2),
                repeat(f_interp),
                repeat(coords),
                repeat(norm),
                repeat(path),
                repeat(bits),
                repeat(fg),
                repeat(bg),
                ),
            total=n))
    else:
        with pool:
            list(
                pool.starmap(
                    makeslice,
                    tqdm(
                        zip(
                            range(n),
                            repeat(z2),
                            repeat(f_interp),
                            repeat(coords),
                            repeat(norm),
                            repeat(path),
                            repeat(bits),
                            repeat(fg),
                            repeat(bg),
                        ), total=n),
                    chunksize=4))


def _sum_imgs(args):
    "helper function for image_sum"
    low, high, files, nx, ny = args
    summed_image = np.zeros([ny, nx])

    for file in files[low:high + 1]:
        summed_image += np.array(imageio.imread(file).sum(-1) < 3 * 255, dtype=int).T
    return summed_image


def image_sum(files, n_proc=None):
    """Sums up the number of filled pixels in the given files

    Parameters
    ----------
    files : list
        list of files to sum up
    n_proc : int, optional
        number of processors to use, by default None

    Returns
    -------
    array
        sum of filled pixels in the given files
    """

    nx, ny = imageio.imread(files[0]).shape[:2]

    # get a pool
    n_proc = n_proc or cpu_count()
    p = Pool(n_proc)

    # prepare the arguments
    r = np.arange(0, len(files), len(files) // n_proc)
    r[-1] = len(files)
    args = list(zip(r[0:], [x - 1 for x in r[1:]], repeat(files), repeat(nx), repeat(ny)))

    return sum(p.map(_sum_imgs, args))


def color_replace(im, orig_color, repl_col, f=[1]):
    """replaces `orig_color` with `repl_col`.

    Several colors can be given, then the keyword `f` will assign the relative frequency"""

    # all pixels with that mask
    color_mask = np.all(im == np.array(orig_color)[None, None, :], -1)

    repl_cols = np.array(repl_col, ndmin=2)
    fs = np.array(f, ndmin=1)

    assert abs(sum(fs)) - 1 < 1e-8, 'the f factors need to sum to 1.'
    assert np.min(fs) >= 0 and np.max(fs) <= 1.0, 'f factors need to be between 0 and 1 (including)'

    # sort in ascending frequency
    i_sort = fs.argsort()
    repl_cols = repl_cols[i_sort, :]
    fs = np.hstack((0.0, np.cumsum(fs[i_sort])))

    n_col = repl_cols.shape[0]

    if n_col == 1:
        im_repl = np.where(color_mask[:, :, None], repl_cols[0], im)
    else:
        rand_idx = np.random.rand(*color_mask.shape)
        im_repl = im.copy()

        for ic in range(n_col):
            im_repl = np.where((
                color_mask &
                (fs[ic] <= rand_idx) &
                (rand_idx <= fs[ic + 1])
            )[:, :, None], repl_cols[ic, :], im_repl)

    return im_repl