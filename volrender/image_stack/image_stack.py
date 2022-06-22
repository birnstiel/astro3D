from pathlib import Path
from itertools import repeat

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from PIL import Image, ImageOps
from tqdm.auto import tqdm


def makeslice(iz, z2, f_interp, coords, norm, path, bits=32, fg=None, bg=None, inverse=False):
    """
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

    inverse : bool
        whether to print out the inverse as well

    bg : array
        background color

    fg : array
        foreground color

    """
    # update coordinates - only last entry changes
    n_y, n_x = coords.shape[:-1]
    copy = coords.copy()
    copy[:, :, -1] = z2[iz]

    # the default foreground and background
    old_fg = np.array([0, 0, 0, 255], dtype=np.uint8)
    old_bg = np.array([255, 255, 255, 255], dtype=np.uint8)
    bg = bg or [0, 0, 0, 255]
    fg = fg or [255, 255, 255, 255]

    # interpolate
    new_layer = f_interp(copy.reshape([-1, 3])).reshape([n_x, n_y]).T

    # normalize, convert to grayscale image
    layer_norm = np.array(norm(new_layer))
    im = Image.fromarray(np.uint8(255 - layer_norm * 255))

    if bits == 1:
        im = im.convert('1')

        if inverse:
            im_inv = im.convert('L')
            im_inv = ImageOps.invert(im_inv)
            im_inv = im_inv.convert('1')
    elif bits == 32:
        im = im.convert('1').convert('RGBA')

        # replace colors
        im_np = np.array(im)
        masks = [np.all(im_np == col, -1) for col in [old_fg, old_bg]]

        for col, mask in zip([fg, bg], masks):
            col = np.array(col, dtype=np.uint8)
            im_np = np.where(mask[:, :, None], col, im_np)
        im = Image.fromarray(im_np.astype(np.uint8))

        if inverse:
            im_inv = im.convert('L')
            im_inv = ImageOps.invert(im_inv)
            im_inv = im_inv.convert('1').convert('RGBA')

    else:
        raise ValueError('bits needs to be 1 or 32')

    # save as image and inverted image
    im.save(path / f'slice_{iz:04d}.png', bits=bits, optimize=True)
    if inverse:
        im_inv.save(path / f'slice_transp_{iz:04d}.png', bits=bits, optimize=True)


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

    n_z = int(height * dpi_z / 2.54)
    n_x = int(n_z / dpi_z * dpi_x)
    n_y = int(n_z / dpi_z * dpi_y)

    n_x += n_x % 2  # add 1 to make it even if it isn't
    n_y += n_y % 2  # add 1 to make it even if it isn't

    x2 = np.linspace(0, data.shape[0] - 1, n_x)
    y2 = np.linspace(0, data.shape[1] - 1, n_y)
    z2 = np.linspace(0, data.shape[2] - 1, n_z)
    coords = np.concatenate((np.meshgrid(x2, y2, z2[0])), axis=-1)

    print(f'  original data: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}')
    print(f'interpoation to: {n_x} x {n_y} x {n_z}')
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
