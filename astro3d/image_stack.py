from pathlib import Path
from itertools import repeat, cycle
import imageio
from colorsys import rgb_to_hsv
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec, font_manager
from matplotlib.colors import LogNorm, Normalize, to_rgb, LinearSegmentedColormap
from scipy.interpolate import interpn
from tqdm.auto import tqdm
from PIL import Image, ImageFont, ImageDraw

from skimage.io import imread_collection

from . import fmodule


def rgb_to_cmyk(color):
    """Converts RGB colors to CMYK

    Parameters
    ----------
    color : array
        RGB(A) values need to be 0 ... 255, A is ignored

    Returns
    -------
    list
        CMYK values
    """
    R, G, B = np.array(color[:3]) / 255

    K = 1 - np.max([R, G, B])
    C = (1 - R - K) / (1 - K)
    M = (1 - G - K) / (1 - K)
    Y = (1 - B - K) / (1 - K)
    return np.array([C, M, Y, K])


def cmyk_to_rgb(CMYK):
    """Converts CMYK colors to RGB

    Parameters
    ----------
    CMYK : array
        CMYK values need to be 0 ... 100

    Returns
    -------
    list
        RGB values, 0 ... 255
    """
    CMYK = np.array(CMYK, ndmin=1)
    return 255 * (1 - CMYK[:3] / 100) * (1 - CMYK[3] / 100)


# define the rigid Vero™ colors
VeroT_sRGB = np.array([255, 255, 255])
VeroC_sRGB = np.array([29, 85, 111])
VeroM_sRGB = np.array([149, 39, 87])
VeroY_sRGB = np.array([192, 183, 52])

# define the RGB values of CMY
C_sRGB = cmyk_to_rgb(np.array([100, 0, 0, 0])).astype(int)
M_sRGB = cmyk_to_rgb(np.array([0, 100, 0, 0])).astype(int)
Y_sRGB = cmyk_to_rgb(np.array([0, 0, 100, 0])).astype(int)

# colors defined in the VoxelPrinting Guide
BaseCyan = np.array([0, 89, 158])
BaseMagenta = np.array([161, 35, 99])
BaseYellow = np.array([213, 178, 0])
BaseBlack = np.array([30, 30, 30])
BaseWhite = np.array([220, 222, 216])


def density2color(iz, z2, f_interp, coords, norm, path, levels=None, sigmas=None, clip=None):
    """
    Computes the color density of one slice of the data (index `iz` within new z-grid `z2`).

    There are two ways:
    - A) a direct mapping of `normalized density -> color density`.
    - B) volumetric contours where the color density is scaled with a Gaussian
         distribution around a given density value.

    Method A)
    ---------

    No kewords need to be given to use that method.

    Method B)
    ---------

    Several regions in density-space can be highlighted, each is defined by its entry in:
    - `levels` = the position in the normalized density (=when `norm` is applied to the
       density) where this color is applied
    - `sigmas` = the width of the gaussian region in normalized density. If a negative value
       is given, it falls back to method A) for just that color.
    - `fill` = the filling factors of the density. If a normalized density value is exactly
       equal to an entry in `levels`, this cell will be always filled if its value in `fill`
       is 1. If it is 0.5, it will be filled with approximately 50% probablility.
       This is used to make the colored regions less opaque.

    Note: the normalized density values are derived by calling `norm(density)` which should
    map the density values to [0 ... 1]. To do the inverse, for example to compute which
    density corresponding to a normalized value of 0.4, one can use the inverse of the norm
    like this: `rho = norm.inverse(0.4)` or `rho = norm.inverse([0.4, 0.6]).data` for an array.

    Note: the coordinates and interpolation function should work like this:
            # coords contains the arrays
            _x, _y, _z = coords
            # z coordinate is replaced every time
            _z = z2[iz]
            # _x, _y, _z are transformed in a (N, 3) array and passed to the interpolation
            new_layer = f_interp(_coords)[:, :, 0].T

    iz : int
        slice index within `z2`

    z2 : array
        the new vertical coordinate array

    f_inter : callable
        the interpolation function taking the coordinates as a (N, 3) array (N points)

    norm : callable
        the normalization function that maps density to 0...1

    path : str | Path | None
        the path into which to store the images, if None, don't write it out

    levels : array
        array of `N_colors` float entries that define the density values of each color

    sigmas : array
        array of `N_colors` floats that describe the density width

    clip : array
        if sigmas is used: after how many sigmas to clip the color density
        if not and clip is of length 2: use those entries as lower, upper clip value

    Returns:

    color_density : np.ndarray
        0.0 ... 1.0 - normalized color density, i.e. how much color should be printed where
    """

    sigmas = np.array(sigmas, ndmin=1)
    if clip is None:
        clip = np.inf

    if path is not None:
        path = Path(path)

    # update coordinates - only last entry changes
    _x, _y, _ = coords
    coords2 = np.array(np.meshgrid(_x, _y, z2[iz])).squeeze().reshape(3, -1).T

    # interpolate: note that we transpose as this is how the image will be safed
    new_layer = f_interp(coords2).reshape(len(_y), len(_x))

    # normalize data
    layer_norm = np.array(norm(new_layer))
    layer_norm[np.isnan(layer_norm)] = 0.0

    # clip if wanted
    if sigmas is None and len(np.array(clip)) == 2:
        layer_norm.clip(min=clip[0], max=clip[1], out=layer_norm)

    if levels is not None:
        # compute the different density contours (but exclude 0.0 which should never be colored)
        dist_sq = (np.array(levels)[None, None, :] - layer_norm[..., None])**2 / (2 * sigmas**2)
        dist_sq[layer_norm == 0.0] = np.inf
        dist_sq[dist_sq > np.array(clip)**2] = np.inf
        color_density = 1. / (1 + dist_sq)

        # the negative sigmas will be replaced with method A)
        color_density[:, :, np.array(sigmas) < 0] = layer_norm[..., None]
    else:
        # we are not using the levels, but the density directly
        color_density = layer_norm

    return color_density


def add_streamlines(coords, z2, iz, layer_dithered, streamlines, radius=None):
    """adds streamlines to a dithered layer as a new material.

    Parameters
    ----------
    coords : tuple
        the coordinates (x, y, z) of the layer
    z2 : array
        the new
    iz : _type_
        _description_
    layer_dithered : array
        the `(nx, ny, n_material)`-sized material assignment for every pixel
        so that `layer_dithered[nx,ny, :] = [0,1,0] would mean that the second of three materials
        is assigned to pixel `(nx, ny)`.
    streamlines : list of arrays
        each element in the list is a `(n,3)` sized array with the 3D positions of `n` points of a streamline.
    radius : float, optional
        with of the streamline, by default 2.5

    Returns
    -------
    layer_dithered
        new layer_dithered that has now the streamlines as extra material
    """

    # begin adding streamlines ---
    _x, _y, _ = coords
    radius = radius or 2.5 * np.diff(_x)[0]
    mask = np.zeros([len(_y), len(_x)], dtype=bool)

    for line in streamlines:
        mask = mask | fmodule.mark_streamline(_x, _y, z2[iz], radius, line).T

    # clear other materials where the mask is true
    # and attach the streamlines as new material
    layer_dithered = np.where(mask[:, :, None], 0, layer_dithered)
    layer_dithered = np.concatenate((layer_dithered, mask[:, :, None]), axis=-1)

    return layer_dithered


def layers2image(layer_dithered, path, iz, colors=None, f=None, bg=255):
    """Convert a (nx, ny, n_materials)-sized layer to an (nx, ny, 3)-sized image

    Parameters
    ----------
    layer_dithered : numpy.ndarray
        the assignments of materials: for each pixel (nx, ny), the
        assigment of materials, for example [0,1] assigns the second
        of two materials.
    path : str | path | None
        output path of the folder where the image files will be stored
        image will not be stored as file if it is `None`, just returned
    iz : integer
        running index that is appended to the file stem
    colors : list | array
        list (or array) or list of lists of RGB colors, by default grayscale will be used.
        for each material (in `n_materials`), either
            - a single color (and an entry of `[1]` in the `f` array), or
            - a list of colors (and mixing ratios in the `f` array) need to be given.

        For example `colors = [[1,1,1], [[1, 0, 0], [0, 1, 0]]]`, and `f=[[1], [0.2, 0.8]]`
        will assign white to the first material and will dither the second material with 20%
        red and 80% green.
    f : list, optional
        mixing ratio for each color, see above

    bg : int, optional
        _description_, by default 1

    Returns
    -------
    np.ndarray
        `(nx, ny, 3)` sized image
    """
    # handle colors and f
    n_level = layer_dithered.shape[-1]
    if colors is None:
        colors = np.linspace(0.0, 1.0, n_level)[:, None] * np.ones(3)[None, :]
        colors = (255 * colors).astype(int)
    if f is None:
        f = np.ones(n_level)

    # now replace the colors
    im = []

    old_colors = np.eye(n_level, dtype=int)

    for col_o, col_n, _f in zip(old_colors, colors, f):
        im += [color_replace(layer_dithered, col_o, col_n, f=_f)]

    # replace the background
    bg = bg * np.ones(3, dtype=int)
    im += [color_replace(layer_dithered, np.zeros(n_level, dtype=int), bg)]

    im = np.array(im).sum(0).astype(np.uint8)

    # save as png
    if path is not None:
        imageio.imwrite(path / f'slice_{iz:04d}.png', im[::-1, :, :])

    return im


def makeslice(iz, z2, f_interp, coords, norm, path,
              levels=None, sigmas=None, fill=1.0,
              colors=None, f=None, bg=255, clip=None,
              streamlines=None, radius=None):
    """
    Prints out one slice of the data (index `iz` within grid `z2`) and stores it as
    a file `slice_{iz:04d}.png` in the folder `path`.

    There are two ways:
    - A) a direct mapping of `normalized density -> color density`.
    - B) volumetric contours where the color density is scaled with a Gaussian
         distribution around a given density value.

    Method A)
    ---------

    No kewords need to be given to use that method.

    Method B)
    ---------

    Several regions in density-space can be highlighted, each is defined by its entry in:
    - `levels` = the position in the normalized density (=when `norm` is applied to the
       density) where this color is applied
    - `sigmas` = the width of the gaussian region in normalized density. If a negative value
       is given, it falls back to method A) for just that color.
    - `fill` = the filling factors of the density. If a normalized density value is exactly
       equal to an entry in `levels`, this cell will be always filled if its value in `fill`
       is 1. If it is 0.5, it will be filled with approximately 50% probablility.
       This is used to make the colored regions less opaque.
    - `colors` the color with which those regions are filled. The color needs to be given as a
       fractional abundance of the materials used in the print. This is usually VeroCyan,
       VeroMagenta, VeroYellow.

    Note: the normalized density values are derived by calling `norm(density)` which should
    map the density values to [0 ... 1]. To do the inverse, for example to compute which
    density corresponds to a normalized value of 0.4, one can use the inverse of the norm
    like this: `rho = norm.inverse(0.4)` or `rho = norm.inverse([0.4, 0.6]).data` for an array.

    Note: the coordinates and interpolation function should work like this:
            # coords contains the arrays
            _x, _y, _z = coords
            # z coordinate is replaced every time
            _z = z2[iz]
            # _x, _y, _z are transformed in a (N, 3) array and passed to the interpolation
            new_layer = f_interp(_coords)[:, :, 0].T

    iz : int
        slice index within `z2`

    z2 : array
        the new vertical coordinate array

    f_inter : callable
        the interpolation function taking the coordinates as a (N, 3) array (N points)

    norm : callable
        the normalization function that maps density to 0...1

    path : str | Path | None
        the path into which to store the images, if None, don't write it out

    levels : array
        array of `N_colors` float entries that define the density values of each color

    sigmas : array
        array of `N_colors` floats that describe the density width

    fill : float | array
        float or array of  `N_colors` floats that describe the filling factor of each color

    clip : array
        after how many sigmas to clip the color density

    colors : list (or list of lists) of colors
        material mixing ratios: each density will be displayed as one color, but each color
        might be a mix of several materials, for example a 50/50 mix of VeroYellow™ and VeroCyan™
        will give a dark green.

        After each density-level is dithered (i.e. three images that have no overlapping filled pixels),
        each density can be translated to a mix of materials to create material-mixed colors.

        Thus every component needs a list of colors, even if there is just one color. If less colors are given,
        then the remaining components are not replaced.

    f : list
        a list of mixing fractions for each color.
        - If `colors` is for example `[[VeroCyan]]`, then `f` should be `[[1.0]]` which means
          that the entire first component is VeroCyan.
        - If `colors` is `[[VeroCyan, VeroMagenta], [VeroCyan]]`, `f` could look like `[[0.2, 0.8], [1.0]]`
        which means that the first component is 20% Cyan, 80% Magenta, and the second component is 100% Cyan.

    bg : array
        The background fill level, default=1 (white)

    streamlines : array
        list of arrays, each array is one streamline, each streamline
        is of shape `(N,3)` where `N` can be different for every streamline. Every
        row gives the x,y,z coordinates of the streamline.

        The streamline color  needs to be given as additional entry in `colors`

    radius : float
        radius of the streamline in data space, defaults to a few cells (2.5 * delta x)

    """

    color_density = density2color(
        iz, z2, f_interp, coords, norm, path,
        levels=levels, sigmas=sigmas, clip=clip)

    # create the dithered images
    layer_dithered = fmodule.dither_colors(color_density * fill).astype(int)

    # add streamlines
    if streamlines is not None:
        layer_dithered = add_streamlines(coords, z2, iz, layer_dithered, streamlines, radius=radius)

    im = layers2image(layer_dithered, path, iz, colors=colors, f=f, bg=bg)

    return layer_dithered, im


def makeslice_from_colordensity(path, color_density, iz, fill=1.0, colors=None, f=None, bg=255):
    """
    Given a normalized color density of shape `(nx, ny, n_materials)`, dither, assign colors, and
    store as a file `slice_{iz:04d}.png` in the folder `path`.

    path : str | Path | None
        the path into which to store the images, if None, don't write it out
    color_density : array
        at each pixel `(nx, ny)` a density value, normalized to 0.0 ... 1.0.
    iz : int
        image is stored in the directory `path` as file `slice_0001.png` if `iz=1`
    fill : float | array
        float or array of  `N_colors` floats that describe the filling factor of each color
    colors : list (or list of lists) of colors
        material mixing ratios: each density will be displayed as one color, but each color
        might be a mix of several materials, for example a 50/50 mix of VeroYellow™ and VeroCyan™
        will give a dark green.

        After each density-level is dithered (i.e. three images that have no overlapping filled pixels),
        each density can be translated to a mix of materials to create material-mixed colors.

        Thus every component needs a list of colors, even if there is just one color. If less colors are given,
        then the remaining components are not replaced.
    f : list
        a list of mixing fractions for each color.
        - If `colors` is for example `[[VeroCyan]]`, then `f` should be `[[1.0]]` which means
          that the entire first component is VeroCyan.
        - If `colors` is `[[VeroCyan, VeroMagenta], [VeroCyan]]`, `f` could look like `[[0.2, 0.8], [1.0]]`
        which means that the first component is 20% Cyan, 80% Magenta, and the second component is 100% Cyan.
    bg : array
        The background fill level, default=1 (white)

    returns
    im : np.ndarray of the image
    """

    # create the dithered images
    layer_dithered = fmodule.dither_colors(color_density * fill)

    im = layers2image(layer_dithered, path, iz, colors=colors, f=f, bg=bg)

    return layer_dithered, im


def process(data, height=10, dpi_x=600, dpi_y=300, dpi_z=941, output_dir='slices',
            norm=None, pool=None, vmin=None, vmax=None, iz=None, x=None, y=None, z=None):
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
    x, y, z : array
        input grids of the data-cube
    output_dir : str, optional
        path of output directory, by default 'slices'
    norm : norm or None or str, optional
        use the given norm,
        or a log norm if set to 'log' or None
        or linear norm if set to 'lin', by default None
    pool : None or pool object, optional
        pool for parallel processing, by default None
    vmin : None or float, optional
        minimum value if norm is not given, by default None
    vmax : None or float, optional
        maximum value if norm is not given, by default None
    iz : None or int or int-array, optional
        if int/int array is given, only this/these slice index/indices are produced, by default None

    Raises
    ------
    ValueError
        if norm is invalid string
    """

    # define the norm

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

        vmax = vmax or norm.vmax or 10**np.ceil(np.log10(data.max()))
        vmin = vmin or norm.vmin or 1e-2 * vmax
        norm = Norm(vmin=vmin, vmax=vmax, clip=True)
    else:
        vmax = vmax or norm.vmax or 10**np.ceil(np.log10(data.max()))
        vmin = vmin or norm.vmin or 1e-2 * vmax

    print(f'{type(norm).__name__} from {vmin:.2g} to {vmax:.2g}')

    # create original grid and interpolation function

    x = x or np.arange(data.shape[0])
    y = y or np.arange(data.shape[1])
    z = z or np.arange(data.shape[2])

    lx = x[-1] - x[0]
    ly = y[-1] - y[0]
    lz = z[-1] - z[0]

    def f_interp(coords):
        return fmodule.interpolate(x, y, z, data, coords)

    # calculate new grids

    n_z = int(dpi_z * height / 2.54)
    n_x = int(n_z * lx / lz / dpi_z * dpi_x)
    n_y = int(n_z * ly / lz / dpi_z * dpi_y)

    n_x += n_x % 2  # add 1 to make it even if it isn't
    n_y += n_y % 2  # add 1 to make it even if it isn't

    x2 = np.linspace(0, data.shape[0] - 1, n_x)
    y2 = np.linspace(0, data.shape[1] - 1, n_y)
    z2 = np.linspace(0, data.shape[2] - 1, n_z)
    coords = (x2, y2, z2)

    print(f'original data: {data.shape[0]} x {data.shape[1]} x {data.shape[2]}')
    print(f'interpoation to: {n_x} x {n_y} x {n_z}')
    print(f'print size: {n_x * 2.54 / dpi_x:.2f} x {n_y * 2.54 / dpi_y:.2f} x {n_z *2.54 / dpi_z:.2f} cm')
    print(f'saving into {output_dir}')
    path = Path(output_dir)

    # make output directory available & clean

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
        print(f'only printing {len(np.array(iz, ndmin=1)) * 2.54 / dpi_z:.2f} cm of it')

    n = len(z2)

    if pool is None:
        list(tqdm(
            map(makeslice, range(n),
                repeat(z2),
                repeat(f_interp),
                repeat(coords),
                repeat(norm),
                repeat(path),
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
                        ), total=n),
                    chunksize=4))


def index_lookup(img, palette):
    """given an RGB image and its color palette, get the
    indexed array (i.e. index within the pallette for each pixel). This is from here:
    stackoverflow.com/questions/72050038
    """
    n = palette.shape[-1]
    M = (1 + palette.max())**np.arange(n)
    p1d, ix = np.unique(palette @ M, return_index=True)
    return ix[np.searchsorted(p1d, img @ M)]


def check_colors(imgs, stride=5, nmax=8):
    """Finds a list of unique colors in the given stack using the fortran module.

    Parameters
    ----------
    imgs : np.ndarray
        image stack, (nx, ny, nz, nc), nc is usually 3 for RGB, 4 for RGBA should work too.
    stride : int, optional
        use only every Nth image, by default 5 which is a bit of a balance of speed and accuracy.
        If you have few narrow points of colors, you might need to really check everything (stride=1)
        If the stack is mostly uniform and checking just a few images is enough, you can go closer to
        stride=nz.
    nmax : int, optional
        how many colors can be detected, by default 8
        we need an estimate of how many colors should be in the image
        as the fortran->python interface f2py cannot do allocatable arrays

    Returns
    -------
    colors, list of colors found
    """
    if imgs.ndim == 3:
        imgs = imgs[:, :, None, :]

    ncol, colors = fmodule.get_colors(imgs[:, :, ::stride, :].astype(int), nmax)

    # this returns the numbers of colors and an array of length nmax
    # so we need to pick
    colors = colors[:ncol, :]

    # we sort them "alphabetically", so white should be on the top
    colors = colors[np.lexsort(np.fliplr(colors).T)][::-1, :]

    return colors


def color_replace(im, orig_color, repl_col, f=[1], inplace=False):
    """replaces `orig_color` with `repl_col` in an image.

    If a list of colors and a list of `f`s are given, then `orig_color``
    is replaced with that mix of colors.

    if `inplace` is `True`, then we are replacing colors in an image (of
    shape of `(nx, ny, 3)`) and could leave the rest of the image as it is.
    This could replace a single color in the image with another color.

    if `inplace` is false, the shape does not need to match. For example the
    input array could be of shape `(nx, ny, 2)`, where a "color" could be `[0, 1]`.
    That colors would then be replaced with the given RGB values such that the output
    image is again of shape `(nx, ny, 3)`.
    """

    # all pixels matching that color with that mask
    color_mask = np.all(im == np.array(orig_color)[None, None, :], -1)

    repl_cols = np.array(repl_col, ndmin=2)
    fs = np.array(f, ndmin=1)

    assert abs(sum(fs)) - 1 < 1e-8, 'the f factors need to sum to 1.'
    assert np.min(fs) >= 0 and np.max(fs) <= 1.0, 'f factors need to be between 0 and 1 (including)'

    if not inplace:
        im = np.zeros([im.shape[0], im.shape[1], len(repl_cols[0])])

    # sort in ascending frequency
    i_sort = fs.argsort()
    repl_cols = repl_cols[i_sort, :]
    fs = np.hstack((0.0, np.cumsum(fs[i_sort])))

    n_col = repl_cols.shape[0]

    if n_col == 1:
        # if there is just one color, we can directly replace that
        im_repl = np.where(color_mask[:, :, None], repl_cols[0, None, None], im)
    else:
        # if a color is to be replaced by a mix of multiple colors,
        # then we do this randomly.
        rand_idx = np.random.rand(*color_mask.shape)
        im_repl = im.copy()

        for ic in range(n_col):
            im_repl = np.where((
                color_mask &
                (fs[ic] <= rand_idx) &
                (rand_idx <= fs[ic + 1])
            )[:, :, None], repl_cols[ic, :], im_repl)

    return im_repl


def show_histogram(data, norm, colors=None, levels=None, sigmas=None, clips=None, f=None, fill=1.0):
    """Shows a histogram of the data and indicates the color levels

    Parameters
    ----------
    data : array
        the data to be shown
    norm : matplotlib.colors.Normalize
        norm that is used to normalized `data` on a [0,1] range
    colors : array-like
        list of colors, shape = `(N_color, 3)`
    levels : array-like
        the normalized density level of each color, shape=`(N_color)`
    sigmas : array-like
        the width around each level in normalized space, shape=`(N_color)`
    clips : array-like
        after how many sigmas should the color be cut off, shape=`(N_color)`
    fill : float | array
        filling factor of each layer
    f : list
        the mixing fractions of each color. 1 by default for a single color.
    """
    if clips is None:
        clips = np.inf * np.ones_like(levels)

    bins = np.linspace(0, 1, 100)
    counts, _ = np.histogram(np.array(norm(data.ravel())), bins=bins)

    if levels is not None:
        if colors is None:
            # get default colors
            mix = np.array([to_rgb(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(levels)]])
        else:
            # mix colors
            if f is None:
                f = [list(np.ones(np.array(col, ndmin=2).shape[0]) / np.array(col, ndmin=2).shape[0]) for col in colors]
            mix = [(np.array(c, ndmin=2) * np.array(_f, ndmin=2).T).sum(0) for c, _f in zip(colors, f)]

    fig, ax = plt.subplots(dpi=150)

    ax.bar(np.array(bins[:-1]), counts, width=np.diff(bins), alpha=0.3)
    ax.step(0.5 * (bins[1:] + bins[:-1]), counts, c='k', lw=1)
    ax.set_xlim(0, 1)
    ax.set_yscale('log')
    ax.set_xlabel('normalized density')
    ax.set_ylim(ax.get_ylim())

    # if we do level-based coloring ....
    if (levels is not None) and (sigmas is not None):

        clips = np.array(clips, ndmin=1)
        sigmas = np.array(sigmas, ndmin=1)
        levels = np.array(levels, ndmin=1)

        for i, (_level, _sig, _clip) in enumerate(zip(levels, sigmas, clips)):
            _col = mix[i]
            if _col.max() > 1:
                _col = _col / 255
            ax.axvline(_level, ls='--', c=_col)

            _y = 10**(np.mean(np.log10(ax.get_ylim())) * (1 + 0.1 * i))

            ax.errorbar(_level, _y,
                        xerr=[
                            [_sig],
                            [_sig]], c=_col, capsize=5)
            ax.errorbar(_level, _y,
                        xerr=[
                            [_clip * _sig],
                            [_clip * _sig]], c=_col, capsize=5, alpha=0.5)

        # estimate the filling factor
        dn = np.array(norm(data.ravel())).reshape(data.shape)
        dist_sq = (np.array(levels)[None, None, :] - dn[..., None])**2 / (2 * sigmas**2)
        dist_sq[dn == 0.0] = np.inf
        dist_sq[dist_sq > np.array(clips)**2] = np.inf
        color_density = 1. / (1 + dist_sq) * fill
        ff = color_density.sum() / np.product(data.shape)
    else:
        ff = (norm(data.ravel())).sum() / np.product(data.shape)

    ax2 = ax.secondary_xaxis('top', functions=(norm.inverse, norm))
    ax2.set_xlabel('original density')

    if type(norm).__name__ == 'Normalize':
        ticks = np.array(norm.inverse([0, 1]))
    elif type(norm).__name__ == 'LogNorm':
        ticks = 10.**np.arange(*np.round(np.log10(np.array(norm.inverse([0, 1])))) + [0, 1])
        ax2.get_xaxis().set_major_locator(ticker.LogLocator())
        ax2.get_xaxis().set_ticks(10.**ticks)
    else:
        raise ValueError('unknown norm type given')

    ax.text(0.05, 0.95, f'approximate filling factor = {ff:.2%}', va='top', transform=ax.transAxes)


def rkstep(x, y, z, vel, p, ds):
    """takes a runge-kutta step

    Parameters
    ----------
    x : array
        regular x grid, shape=(nx)
    y : array
        regular y grid, shape=(ny)
    z : array
        regular z grid, shape=(nz)
    vel : array
        velocity/vector field, shape = (nx, ny, nz, 3)
    p : array
        starting point, shape = (3)
    ds : float
        length of the step

    Returns
    -------
    array
        next position, shape=(3)
    """
    pdot0 = interpn((x, y, z), vel, p, bounds_error=False, fill_value=0.0)[0]

    v = np.sqrt(sum(pdot0**2))
    if (v == 0.0):
        return p

    dt = ds / v

    pdot1 = interpn((x, y, z), vel, p + dt / 2.0 * pdot0, bounds_error=False, fill_value=0.0)[0]

    v = np.sqrt(sum(pdot1**2))
    if (v == 0.0):
        return p

    # update the position
    dt = ds / v
    p = p + dt * pdot1
    return p


def streamline(x, y, z, vel, p, length=1.0, n_steps=50):
    """computed a streamline from point p

    Parameters
    ----------
    x : array
        regular x grid, shape = (nx)
    y : array
        regular y grid, shape = (ny)
    z : array
        regular z grid, shape = (nz)
    vel : array
        vector field, shape = (nx, ny, nz, 3)
    p : array
        initial position, shape (3)
    length : float, optional
        approximate path length, by default 1.0
    n_steps : int, optional
        number of steps along the path, by default 50

    Returns
    -------
    array
        the path strating at p, shape = (nsteps, 3)
    """
    path = p * np.ones([n_steps, 3])

    ds = length / n_steps

    # go forward one length

    for i in range(2, n_steps):
        path[i, :] = rkstep(x, y, z, vel, path[i - 1, :], ds)

    return path


def dither_palette(img, pal, resize=None):
    """Dither an image to a given pallette and resizes it to the given number of pixels.
    If an alpha channel is provided, this will be rescaled as well.

    Parameters
    ----------
    img : array
        RGB array of size nx, ny, 3 or 4
    pal : list | array
        color palette, list of RGB values
    resize : tuple, optional
        new shape of image, 2-element tuple, by default None

    Returns
    -------
    array
        dithered RGBA image

    """
    pal = np.array(pal, ndmin=2)
    if pal.shape[-1] < 3:
        raise ValueError('palette needs to be RGB colors')
    else:
        pal = pal[:, :3]

    img = np.array(img, ndmin=3)
    if img.shape[-1] not in [3, 4]:
        raise ValueError('image needs to be RGB or RGBA colors')

    # resize
    im1 = Image.fromarray(img).convert('RGBA')

    if resize is not None:
        im1 = im1.resize(tuple(list(resize)[::-1]))

    alpha_mask = np.array(im1)[:, :, -1]

    # quantize it
    p_img = Image.new('P', (1, 1))
    p_img.putpalette(list(pal.ravel()))
    im1 = im1.convert('RGB').quantize(palette=p_img).convert('RGB')
    im1 = np.array(im1)

    # return as array
    im_out = np.zeros([*im1.shape[:2], 4], dtype=np.uint8)
    im_out[:, :, :3] = im1
    im_out[:, :, 3] = alpha_mask

    return im_out


def _get_text_image(text, size=100, family='sans-serif', weight='regular', bg=255):
    """Creates a black/white image of the given text.

    Parameters
    ----------
    text : str
        string to be printed in the image
    size : int, optional
        font size, by default 100
    family : str, optional
        font family, by default 'sans-serif'
    weight : str, optional
        font weight, by default 'regular'
    bg : int
        background brightness, white by default 255

    Returns
    -------
    numpy.ndarray
        2D array data where the font will have 0, the background have 255 by default
    """
    color = (0, 0, 0)
    font = font_manager.FontProperties(family=family, weight=weight)
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, size=size)

    bbox = font.getbbox(text)

    # create empty image
    img = 255 * np.ones([bbox[3], bbox[2], 3], dtype=np.uint8)
    img = Image.fromarray(img)

    # write text into it
    img_edit = ImageDraw.Draw(img)
    img_edit.text((0, 0), text, color, font=font)

    # convert back to numpy
    im_out = np.asarray(img)

    # make it "binary"
    im_out = (im_out.mean(-1) > 128) * bg

    im_out = im_out.T[:, ::-1]

    return im_out


def _convert_image(image, width, dx, dy, height=None, threshold=None, pal=None):
    """converts an image file to a grid that matches an image stack.

    This will usually dither the image using the palette `pal` or a default palette.
    Passing `threshold` value will avoid dithering, and use the brightness level to
    separate into background/foreground. This will then use the first two entries
    in `pal` to assign foreground & background.

    If the input image is RGBA, the second return value will be a reshaped alpha mask
    to try and retain the transparency of the image.

    Parameters
    ----------
    image : str | pathlib.Path | np.ndarray | Image
        path to the image file or array with image data
        if image data is 3D (shape=(nx, ny, 3 or 4)), it will be converted as an image (origin='lower')
        if image data is 2D, it will be treated as array (first element = bottom left)
    width : float
        width of the final logo in cm. Can be set to `None` if `height` is given (keeping the aspect ratio).
    dx : float
        voxel width
    dy : float
        voxel heigh
    height : float, optional
        height of the final log in cm, by default (`None`) the aspect ratio is kept. If both `width` and
        `height` are given, this will override the image aspect ratio.
    threshold : None | float
        if `float`, the image will be converted to black/white with the given brightness threshold (0.0 ... 1.0).
        The the low (high) value will be filled with the first (second) entry of `pal`. Default Black & White
    pal : list | array, optional
        color palette, list of RGB values, by default None

    Returns
    -------
    array:
        an RBG image with limited palette of colors and rescaled to match the dimensions of the image stack.

    array:
        an alpha_mask that can be applied to the image to remove background on the rescaled image.
    """

    # we use the the 3 colors + black and white as default
    if pal is None:
        pal = [BaseBlack, BaseWhite, BaseCyan, BaseMagenta, BaseYellow]

    # if path given, read image
    if isinstance(image, (str, Path)):
        # Read image
        image = imageio.v3.imread(image)

    # if input is image-like, rotate
    if isinstance(image, (imageio.core.Array, Image.Image)) or (isinstance(image, np.ndarray) and image.ndim in [3, 4]):
        im = np.rot90(np.array(image), axes=(1, 0))
    # if input is 2D array, use as is, but add 3rd dimension
    elif isinstance(image, np.ndarray) and image.ndim == 2:
        im = np.zeros([*image.shape, 3], dtype=np.uint8)
        im[...] = image[:, :, None]
    else:
        raise ValueError('Input needs to be image path, image, or array')

    if width is None and height is None:
        width = dx * im.shape[0]
        height = dy * im.shape[1]

    # if height not given: keep aspect ratio
    if height is None:
        height = width / im.shape[0] * im.shape[1]
    if width is None:
        width = height * im.shape[0] / im.shape[1]

    # new shape: same extent with right number of steps
    nx = int(np.ceil(width / dx))
    ny = int(np.ceil(height / dy))

    if threshold is None:
        # DITHERING
        im = dither_palette(im, pal, resize=(nx, ny))
    else:
        # reshape without dithering
        im = Image.fromarray(im).convert('RGBA')
        im = np.array(im.resize((ny, nx)))

        # make the alpha channel binary
        alpha_mask = np.where(im[:, :, 3] > 0, 255, 0).astype(np.uint8)

        # make the image binary
        mask = (im[:, :, :3].mean(-1) / 255) < threshold
        im[:, :, :3] = np.where(mask[:, :, None], pal[0], pal[1])
        im[:, :, 3] = alpha_mask

    return im


class IStack(object):
    def __init__(self, input, dpi_x=600, dpi_y=300, dz=27e-4):
        """Image stack for 3D printing.

        Parameters
        ----------
        input : str | Path | numpy.ndarray
            input can be a directory with image files or a 3D numpy array
        dpi_x : float
            x-resolution in DPI
        dpi_y : float
            y-resolution in DPI
        dz : float
            z-layer width in cm

        Raises
        ------
        ValueError
            if input is not a string, a directory path, or a numpy ndarray
        """

        self.directory = None
        self.files = None

        # set printer properties
        self.dx = 2.54 / dpi_x
        self.dy = 2.54 / dpi_y
        self.dz = dz

        self.dpi_x = dpi_x
        self.dpi_y = dpi_y
        self.dpi_z = 2.54 / dz

        if isinstance(input, np.ndarray):
            self.imgs = input
        #  input can be directory
        elif isinstance(input, (Path, str)):
            self.directory = Path(input)
            if not self.directory.is_dir():
                raise ValueError('intput has to be a directory or numpy array')

            self.files = sorted(list(self.directory.glob('*.png')))

            # read image stack - the transpose makes the order x, y, z
            collection = imread_collection(str(self.directory / '*.png'))
            self._imgs = collection.concatenate()
            self.imgs = self._imgs.transpose(2, 1, 0, 3)[:, ::-1, ...]
        else:
            raise ValueError('input has to be directory or numpy ndarray')

        # we make it unwritable so that it can only changed via the property, or
        # within functions, where we need to call the update() function to reset
        # some things
        self._imgs.flags.writeable = False

        # determine the palette
        self._colors = None
        self._ncol = None
        self._counts = None

        # which indices are empty
        self._empty_indices = None

    @property
    def imgs(self):
        return self._imgs

    @imgs.setter
    def imgs(self, value):
        # set the value
        if not isinstance(value, np.ndarray) or value.ndim != 4:
            raise ValueError('you need to give a 4D numpy.ndarray')
        self._imgs = value
        self.reset()

    def reset(self):
        """resets simple thing in case the data changed"""
        self.nx, self.ny, self.nz = self._imgs.shape[:3]
        self._x = np.arange(self.nx) * self.dx
        self._y = np.arange(self.ny) * self.dy
        self._z = np.arange(self.nz) * self.dz

        # reset dependent properties that need longer to calculate
        self._colors = None
        self._ncol = None
        self._counts = None
        self._get_view.cache_clear()

    @property
    def empty_indices(self):
        "the indices in the color palette that are empty (no material or transparent)"
        if self._empty_indices is None:
            self._empty_indices = []
        return self._empty_indices

    @empty_indices.setter
    def empty_indices(self, value):
        value = np.array(value, ndmin=1)
        value[value == -1] = self.ncol - 1
        if value.min() < 0 or (value.max() > self.ncol - 1):
            raise ValueError('indices outside of color palette length')
        self._empty_indices = value

    @property
    def nonempty_indices(self):
        "the indices in the color palette that are not empty (=not transparent)"
        return self._get_nonempty(self.empty_indices)

    def _get_nonempty(self, empty_indices):
        return [i for i in range(self.ncol) if i not in empty_indices]

    @property
    def colors(self):
        if self._colors is None:
            self._get_colors()
        return self._colors

    @property
    def ncol(self):
        if self._colors is None:
            self._get_colors()
        return len(self.colors)

    def _get_colors(self, stride=5):
        """Get all colors present in a random sub-sample of N slices.
        """
        self._colors = check_colors(self.imgs, nmax=8, stride=stride)
        self._ncol = len(self._colors)
        print('Done!')

    @property
    def counts(self):
        """the number of pixels filled in each column for each material"""
        if self._counts is None:

            # add up the occurences of all colors
            self._counts = np.zeros([self.nx, self.ny, self.ncol])
            idx = np.arange(len(self.colors))
            for i in tqdm(range(self.nz), desc='computing counts'):
                im_idx = index_lookup(self.imgs[:, :, i, :], self.colors)
                self._counts += (im_idx[:, :, None] == idx[None, None, :])
        return self._counts

    def show_colors(self, titles=[], **kwargs):
        """Shows a figure with all colors

        Parameters
        ----------
        titles : list, optional
            list of color names, by default []

        other keywords are passedto the `Axes.Text` call

        Returns
        -------
        f, ax
            figure and axes handles
        """
        f = plt.figure(figsize=(len(self.colors), 1))
        ax = f.add_axes([0, 0, 1, 1])
        ax.imshow([self.colors])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if 'ha' not in kwargs or 'horizontalalignment' not in kwargs:
            kwargs['ha'] = 'center'

        if 'color' not in kwargs or 'c' not in kwargs:
            tcolors = [str(round(1 - rgb_to_hsv(*col)[-1] / 255)) for col in self.colors]
        else:
            tcolors = kwargs.pop('color', kwargs.pop('c', None))

        if len(titles) > 0:

            for i, title, tcol in zip(range(len(self.colors)), cycle(titles), cycle(tcolors)):
                ax.text(i, -0.35, title, c=tcol, **kwargs)
        return f, ax

    def replace_color(self, i_col, new_col):
        """Replaces the color of index `i_col` with the new color `new_col`."""
        new_col = np.array(new_col, dtype=self.imgs.dtype)
        mask = (self.imgs == self.colors[i_col][None, None, None, :]).all(-1)

        # make it writable briefly
        # but here we update the colors manually to save time
        self._imgs.flags.writeable = True
        self.imgs = np.where(mask[:, :, :, None], new_col[None, None, None, :], self.imgs)
        self._imgs.flags.writeable = False

    def replace_color_mix(self, i_col, repl_colors, f=[1]):
        """Replaces the color at index `i_col` with a *mix* of new colors.

        This is slower than `replace_color`, but can randomly mix different colors
        with mixing ratios defined in `f`.

        Parameters
        ----------
        i_col : int
            index of the color to be replaces within self.colors
        repl_colors : list of colors
            the colors that should replace the original color. the relative
            abundance of each is given by `f`.
        f : list of floats, optional
            the mixing ratios of the colors, should add to 1, by default [1]
        """
        if len(f) != len(repl_colors):
            raise ValueError('Each new color needs a relative abundance given in `f`.')

        for iz in range(self.nz):
            self._imgs.flags.writeable = True
            self.imgs[:, :, iz, :] = color_replace(
                self.imgs[:, :, iz, :],
                self.colors[i_col],
                repl_colors,
                f=f,
                inplace=True)
            self._imgs.flags.writeable = False
        # reset as colors might have changed
        self.reset()

    def show_histogram(self, empty_indices=None):
        """Shows a histogram of all non-transparent materials.

        Parameters
        ----------
        empty_indices : list, optional
            indices within colors that count as empty (void or transparent), defaults to class attribute `empty_indices`

        Returns
        -------
        figure, axes
            figure and axes objects
        """
        empty_indices = empty_indices or self.empty_indices
        rest = self._get_nonempty(empty_indices)

        f, ax = plt.subplots()
        _ = ax.hist(self.counts[:, :, rest].ravel(), bins=np.linspace(0, 20, 21) - 0.5)
        ax.set_yscale('log')
        ax.set_xlim(right=20)
        return f, ax

    def show_counts(self, vmax=10, empty_indices=None):
        """display the distribution of colors along the vertical dimension.

        Parameters
        ----------
        vmax : int, optional
            upper end of the color scale, by default 10

        empty_indices : array, optional
            which indices to skip, defaults to attribute self.empty_indices

        Returns
        -------
        figure, axis
            returns the figure and axis object
        """
        empty_indices = empty_indices or self.empty_indices

        n_columns = self.ncol - len(empty_indices)

        f, ax = plt.subplots(2, n_columns, figsize=(5 * n_columns, 5), gridspec_kw={'height_ratios': [1, 20]}, dpi=150)

        ax = np.array(ax, ndmin=2)
        if n_columns == 1:
            ax = ax.T

        i_column = -1

        for ic, col in enumerate(self.colors):
            # skip the transparent regions
            if ic in self.empty_indices:
                continue
            i_column += 1

            # assign a color map from white to that color
            # for colors close to white, we invert it
            if np.any(col > 1.):
                col = col / 255
            if (col).sum() > 2.8:
                bg = [0., 0., 0.]
            else:
                bg = [1., 1., 1.]
            cmap = LinearSegmentedColormap.from_list('my', [bg, col])

            # plotting
            cc = ax[1, i_column].imshow(self.counts[:, :, ic].T, vmin=0, vmax=vmax, origin='lower', cmap=cmap)
            ax[1, i_column].set_aspect(self.dpi_x / self.dpi_y)
            f.colorbar(cc, cax=ax[0, i_column], orientation='horizontal')

            ax[1, i_column].set_xlabel('x [voxel]')
            ax[1, i_column].set_ylabel('y [voxel]')

        return f, ax

    @property
    def dimension(self):
        "[x, y, z] size in cm"
        return [self.nx * self.dx, self.ny * self.dy, self.nz * self.dz]

    def show_info(self, empty_indices=None):
        "prints some infos about the image stack (size, filling factor, ...)"
        # indices of the non-transparent
        empty_indices = empty_indices or self.empty_indices
        rest = self._get_nonempty(empty_indices)

        print(f'There are {self.ncol} colors in this image:')
        for i, row in enumerate(self.colors):
            print(f'- {list(row)}' + (i in empty_indices) * ' (transp.)')

        dim = self.dimension

        print(f'{self.nz} files')
        print(f'dimension = {dim[0]:.2f} x {dim[1]:.2f} x {dim[2]:.2f} cm')
        print(f'filling fraction: {1 - self.counts[:, :, empty_indices].sum() / (self.nx * self.ny * self.nz):.2%}')

        print(f'nr of fully transparent columns: {(self.counts[:, :, rest].sum(-1) == 0).sum() / (self.nx * self.ny * self.nz):.2%}')
        print(f'most opaque pixel has {self.counts[:, :, rest].sum(-1).max():n} filled pixels (={self.counts[:, :, rest].sum(-1).max() / self.nz:.2%} of all layers are filled)')

        print('mean counts in non-transparent columns: ' + ', '.join([f'{np.mean(self.counts[:, :, i][self.counts[:,:,i]!=0]):.2g}' for i in range(self.ncol)]))

    def show_transparency_estimate(self, empty_indices=None):
        "shows a map of the estimated projected transparency"

        empty_indices = empty_indices or self.empty_indices
        rest = self._get_nonempty(empty_indices)

        f, axs = plt.subplots(1, 3, figsize=(11, 3), dpi=100, tight_layout=True)

        opts = dict(cmap='gray', origin='lower')
        summed_image = self.counts[:, :, rest].sum(-1)

        ax = axs[0]
        cc = ax.imshow(summed_image.T, **opts, vmin=0, vmax=7)
        ax.set_aspect(self.dpi_x / self.dpi_y)
        plt.colorbar(cc, ax=ax).set_label('# of opaque pixes along LOS')

        ax = axs[1]
        i = ax.imshow(summed_image.T, vmin=0, vmax=summed_image.max(), **opts)
        ax.set_aspect(self.dpi_x / self.dpi_y)
        plt.colorbar(i, ax=ax).set_label('# of opaque pixes along LOS')

        ax = axs[-1]
        counts, bins, patches = ax.hist(summed_image.ravel(), bins=np.arange(summed_image.max()))
        ax.set_yscale('log')
        ax.set_xlabel('number of filled voxels in column')
        ax.set_xlabel('count')
        return f, ax

    @lru_cache(maxsize=12)
    def _get_view(self, n_tauone=7, bg=(255, 255, 255), view='xy', backward=False):
        """computes an approximate render view from bottom, assuming the optical depth is around `n_tauone` pixels.

        Parameters
        ----------
        n_tauone : int, optional
            after how many pixels the print becomes optically thick, by default 7

        bg : array, optional
            background image color, default white, i.e. [255, 255, 255]

        view : str
            viewing direction: 'xy' (default), 'xz', 'yz'

        backward : bool
            view from opposite direction, default: looking in ascending direction

        Returns
        -------
        image : array
            rendered image
        extent: list
            extent of the image
        aspect: float
            aspect ratio for plotting image real-world aspect
        """

        bg = (np.ones(3) * bg).astype(int)

        if view not in ['xy', 'xz', 'yz']:
            raise ValueError('invalid view')

        if view == 'xy':
            data = self.imgs
        elif view == 'xz':
            data = self.imgs.transpose(0, 2, 1, 3)
        elif view == 'yz':
            data = self.imgs.transpose(1, 2, 0, 3)

        x = [0, data.shape[0]]
        y = [0, data.shape[1]]

        aspect = getattr(self, f'dpi_{view[0]}') / getattr(self, f'dpi_{view[1]}')

        # note, the forward integration is the ray to observer
        # so in xy, backward=False is the view from top.
        # further more, we pass the indices to fortran, so we
        # need to use fortran indexing (1, n)
        if backward:
            i0 = 1
            i1 = data.shape[2]
            step = 1
        else:
            i0 = data.shape[2]
            i1 = 1
            step = -1

        image = fmodule.compute_view(data, i0, i1, step, n_tauone, self.colors[self.empty_indices], bg=bg)

        if view == 'xy':
            if not backward:
                y = y[::-1]
                image = image[:, ::-1]
        elif view == 'xz':
            if backward:
                x = x[::-1]
                image = image[::-1, :]
        elif view == 'yz':
            if not backward:
                x = x[::-1]
                image = image[::-1, :]

        extent = [*x, *y]

        return image, extent, aspect

    def show_view(self, view, bg=(255, 255, 255), n_tauone=7, backward=False, ax=None, **kwargs):
        """plots an approximate render view from bottom, assuming the optical depth is around `n_tauone` pixels.

        Parameters
        ----------

        view : str
            viewing direction: 'xy' (default), 'xz', 'yz'

        n_tauone : int, optional
            after how many pixels the print becomes optically thick, by default 7, for colors 10 or so is better

        bg : array, optional
            background image color, default white, i.e. [255, 255, 255]

        backward : bool
            view from opposite direction, default: looking in ascending direction

        ax : matplotlib axes
            into which axes to plot, will create new figure/axes if None

        Returns
        -------
        figure, axes of the rendered view
        """

        bg = tuple((np.ones(3) * bg).astype(int))

        image, extent, aspect = self._get_view(n_tauone=n_tauone, bg=bg, view=view, backward=backward)

        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.figure

        ax.imshow(image.transpose(1, 0, 2).astype(int), extent=extent, origin='lower')
        ax.set_aspect(aspect)
        ax.set_xlabel(view[0] + '-axis')
        ax.set_ylabel(view[1] + '-axis')

        return f, ax

    def show_all_sides(self):
        "Make a plot of all 6 sides (takes some time!)"
        f, ax = plt.subplots(3, 2, figsize=(8, 10))

        for ix, backward in enumerate([True, False]):
            for iy, view in enumerate(['xy', 'xz', 'yz']):
                image, extent, aspect = self._get_view(view=view, backward=backward)
                ax[iy, ix].imshow(image.transpose(1, 0, 2).astype(int), extent=extent, origin='lower')
                ax[iy, ix].set_aspect(aspect)
                ax[iy, ix].set_title(f'backward = {backward}')
                ax[iy, 0].set_xlabel(view[0])
                ax[iy, 0].set_ylabel(view[1])
        return f, ax

    def three_views(self, bg=[255] * 3):

        # convert to tuple
        bg = tuple((np.ones(3) * bg).astype(int))

        f = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[1, self.imgs.shape[2] / self.dpi_z / (self.imgs.shape[1] / self.dpi_y)],
            height_ratios=[1, self.imgs.shape[2] / self.dpi_z / (self.imgs.shape[0] / self.dpi_x)])
        gs.update(wspace=0.0, hspace=0.0)

        ax1 = plt.subplot(gs[0, 0])
        self.show_view(bg=bg, view='xy', ax=ax1, backward=True)
        ax1.set_xticks([])
        ax1.text(0.02, 0.99, 'top', color='red', ha='left', va='top', transform=ax1.transAxes)

        ax3 = plt.subplot(gs[1, 0])
        self.show_view(bg=bg, view='xz', ax=ax3)
        ax3.text(0.02, 0.99, 'front', color='red', ha='left', va='top', transform=ax3.transAxes)

        # special treatment to rotate right figure
        ax2 = plt.subplot(gs[0, 1])
        image, extent, aspect = self._get_view(bg=bg, view='yz', backward=True)
        ax2.imshow(image[:, ::-1].astype(int), extent=np.array(extent)[[3, 2, 0, 1]], origin='lower')
        ax2.set_aspect(1 / aspect)
        ax2.set_xlabel('z-axis')
        ax2.set_ylabel('y-axis')
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax2.text(0.02, 0.99, 'right', color='red', ha='left', va='top', transform=ax2.transAxes)

        return f, [ax1, ax2, ax3]

    def add_shell(self, thickness, level=200,
                  bottom=True,
                  top=True,
                  left=True,
                  right=True,
                  front=True,
                  back=True):
        """adds a constant width shell around the image cube.

        Parameters
        ----------
        thickness : float
            shell thickness in cm
        level : int, optional
            grey scale value of the shell material, by default 200
        bottom, top, left, right, front, back : bool, optional
            whether or not to add the bottom, top, ... side of the shell, by default True
        """

        n_x = int(thickness / 2.54 * self.dpi_x)
        n_y = int(thickness / 2.54 * self.dpi_y)
        n_z = int(thickness / 2.54 * self.dpi_z)

        self.imgs = np.pad(self.imgs, (
            (left * n_x, right * n_x),
            (front * n_y, back * n_y),
            (bottom * n_z, top * n_z),
            (0, 0)),
            mode='constant', constant_values=level)

        self.nx, self.ny, self.nz = self.imgs.shape[:-1]

        self._counts = None

    def add_logo(self, fname, pos, width, depth, plane='xz',
                 height=None, flip_x=False, flip_y=False, pal=None, threshold=None):
        """adds a logo to the image stack based on an image file

        Parameters
        ----------
        fname : str | pathlib.Path | Image | array
            file name of the logo image or an image object / array
        pos : array
            position of the logo in x, y, z in units of cm
        width : float
            logo width in cm. Can be set to `None` if height is given instead.
            Then `height` and keeping the aspect ratio will determine the width.
        depth : float
            depth of the image in cm (how thick it is). We recommend something like
            7-10 * dx to make it optically thick.
        plane : str
            in which plane the image should be printed: 'xy', 'yz', or 'xz' (default)
        height : float, optional
            height of logo, by default None which will keep the aspect ratio given `width`
        flip_x, flip_y : bool
            if the logo should be flipped in (its) x or y direction
        pal : None | array
            palette to use for dithering the logo. Will default to the `Base` color pallete
            specified by Stratasys.
        threshold : None | float
            If the image should not be dithered, but a brightness threshold (0.0 ... 1.0)
            should be used instead to make a binary image, pass this here. For an image
            made of a single print material, this gets rid of the dithering noise.
            The the low (high) brightness value will be filled with the first (second) entry of `pal`
        """
        if plane not in ['xy', 'yz', 'xz']:
            raise ValueError('plane does not exist')

        if pal is None:
            pal = [BaseBlack, BaseWhite, BaseCyan, BaseMagenta, BaseYellow]

        # convert image
        im2 = _convert_image(fname, width,
                             getattr(self, 'd' + plane[0]),  # e.g. self.dx
                             getattr(self, 'd' + plane[1]),  # e.g. self.dy
                             height=height, pal=pal, threshold=threshold)

        alpha_mask = im2[:, :, 3]
        im2 = im2[:, :, :3]

        if flip_x:
            im2 = im2[::-1, :]
        if flip_y:
            im2 = im2[:, ::-1]

        # convert position from cm to index
        p0 = (pos / np.array([self.dx, self.dy, self.dz])).astype(int)

        if plane == 'xy':
            ndepth = round(depth / self.dz)
            im_size = [im2.shape[0], im2.shape[1], ndepth]
            im2 = im2[:, :, None, :]
            alpha_mask = alpha_mask[:, :, None, None]
        elif plane == 'yz':
            ndepth = round(depth / self.dx)
            im_size = [ndepth, im2.shape[0], im2.shape[1]]
            im2 = im2[None, :, :, :]
            alpha_mask = alpha_mask[None, :, :, None]
        elif plane == 'xz':
            ndepth = round(depth / self.dy)
            im_size = [im2.shape[0], ndepth, im2.shape[1]]
            im2 = im2[:, None, :, :]
            alpha_mask = alpha_mask[:, None, :, None]

        if any(p0 + im_size > self.imgs.shape[:-1]):
            raise ValueError(f'image block of size {im_size} put at {p0} exceeds stack size {self.imgs.shape[:-1]}')

        self._imgs.flags.writeable = True

        # define a matching-size view into the stack
        slice = self.imgs[p0[0]:p0[0] + im_size[0], p0[1]:p0[1] + im_size[1], p0[2]:p0[2] + im_size[2]]

        # assign this array to the original image stack
        slice[...] = np.where(alpha_mask == 255, im2, slice)
        self._imgs.flags.writeable = False
        self.reset()

    def add_streamlines(self, streamlines, radius=None, x=None, y=None, z=None, color=[0, 0, 0]):
        """adds streamlines to the image stack.

        Parameters
        ----------
        x, y, z : array
            the coordinates (x, y, z) of the slice, defaults to positoin within the image stack (in cm)
        streamlines : list of arrays
            each element in the list is a `(n,3)` sized array with the 3D positions of `n` points of a streamline.
        radius : float, optional
            with of the streamline, by default 2.5 cells in x-direction
        color : 3-element array
            color that will used for the streamline

        """

        if x is None:
            x = np.arange(self.imgs.shape[0]) * self.dx
        if y is None:
            y = np.arange(self.imgs.shape[1]) * self.dy
        if z is None:
            z = np.arange(self.imgs.shape[2]) * self.dz

        radius = radius or 2.5 * self.dx

        color = np.array(color, ndmin=1)
        if color.ndim != 1 and len(color) != 3:
            raise ValueError('color needs to have 3 entries.')

        zmin = min([np.array(streamline[:, 2]).min() for streamline in streamlines]) - radius
        zmax = max([np.array(streamline[:, 2]).max() for streamline in streamlines]) + radius

        # make image stack writeable
        self._imgs.flags.writeable = True
        try:
            for iz in tqdm(range(self.imgs.shape[2])):
                if (z[iz] < zmin) or (z[iz] > zmax):
                    continue

                mask = np.zeros([len(x), len(y)], dtype=bool)
                for line in streamlines:
                    mask = mask | fmodule.mark_streamline(x, y, z[iz], radius, line)

                self.imgs[:, :, iz, :] = np.where(mask[:, :, None], color[None, None, :], self.imgs[:, :, iz, :])
        finally:
            self._imgs.flags.writeable = False

        self.reset()

    def add_scale(self, L, p0, plane='xz', bar_ratio=0.2, color=[255, 0, 0], radius=None):
        """add a scale bar

        Parameters
        ----------
        L : float
            length of bar in cm
        p0 : array
            x, y, z position in cm
        plane : str, optional
            which plane of 'xy', 'yz', 'xz', by default 'xz'
        bar_ratio : float, optional
            ratio of bar height by bar length, by default 0.2
        color : list, optional
            color of the line, by default [255, 0, 0]
        radius : float, optional
            radius passed to `add_streamline`

        Raises
        ------
        ValueError
            if plane does not exist
        """
        if plane == 'xy':
            vec_x = np.array([1, 0, 0])
            vec_y = np.array([0, 1, 0])
        elif plane == 'yz':
            vec_x = np.array([0, 1, 0])
            vec_y = np.array([0, 0, 1])
        elif plane == 'xz':
            vec_x = np.array([1, 0, 0])
            vec_y = np.array([0, 0, 1])
        else:
            raise ValueError('plane needs to be xy, yz, xz')

        p0 = np.array(p0, ndmin=1)

        bar = np.array([
            [p0, p0 + L * vec_x],
            [p0 + bar_ratio / 2 * L * vec_y, p0 - bar_ratio / 2 * L * vec_y],
            [p0 + L * vec_x + bar_ratio / 2 * L * vec_y, p0 + L * vec_x - bar_ratio / 2 * L * vec_y]])

        self.add_streamlines(bar, color=color, radius=radius)

    def add_box(self, pos, length=1.0, color=[255, 0, 0]):
        """draw the sides of a cube: two squares, and 4 lines connecting them

        Parameters
        ----------
        pos : 3-element array
            (x, y, z) position of the box
        length : float or array, optional
            side-length of the cube, by default 1.0, can be array (lx, ly, lz)
        color : list, optional
            color of the box, by default [255, 0, 0]
        """
        length = length * np.ones(3)
        path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        vert = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1]])
        box = [
            path,
            path + [0, 0, 1],
            vert,
            vert + [0, 1, 0],
            vert + [1, 0, 0],
            vert + [1, 1, 0]]

        box = [length * b + pos for b in box]

        # add box
        self.add_streamlines(box, color=color)

    def add_sphere(self, pos, radius=0.1, color=[255, 0, 0]):
        """draw a filled sphere of given radius centered around `pos`

        Parameters
        ----------
        pos : array | list
            (x, y, z) position of the center
        radius : float, optional
            radius in cm, by default 0.1
        color : list, optional
            color of the sphere, by default [255, 0, 0]
        """
        ix = self._x.searchsorted(pos[0])
        iy = self._y.searchsorted(pos[1])
        iz = self._z.searchsorted(pos[2])

        nx = int(np.ceil(radius / self.dx))
        ny = int(np.ceil(radius / self.dy))
        nz = int(np.ceil(radius / self.dz))

        x, y, z = np.meshgrid(
            self._x[ix - nx:ix + nx + 1],
            self._y[iy - ny:iy + ny + 1],
            self._z[iz - nz:iz + nz + 1], indexing='ij')
        mask = ((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2) < radius**2

        color = np.array(color, ndmin=1)

        try:
            self._imgs.flags.writeable = True
            slice = self._imgs[
                ix - nx:ix + nx + 1,
                iy - ny:iy + ny + 1,
                iz - nz:iz + nz + 1, :]
            res = np.where(mask[:, :, :, None], color[None, None, None, :], slice)
            slice[...] = res
        finally:
            self._imgs.flags.writeable = False

        self.reset()

    def add_text(self, text, pos, width, depth, plane='xz', height=None, col=[0, 0, 0],
                 flip_x=False, flip_y=False, size=100, family='sans-serif', weight='regular', bg=255):
        """Adds text to the image stack

        Parameters
        ----------
        text : str
            string to be printed in the image
        pos : array
            position of the logo in x, y, z in units of cm
        width : float
            logo width in cm
        depth : float
            depth of the image in cm

        plane : str
            in which plane the image should be printed: 'xy', 'yz', or 'xz' (default)
        height : float, optional
            height of logo, by default None
        col : list, optional
            color to be assigned at dark parts of logo, by default [0, 0, 0]
        flip_x, flip_y : bool
            if the logo should be flipped in (its) x or y direction
        size : int, optional
            font size, by default 100
        family : str, optional
            font family, by default 'sans-serif'
        weight : str, optional
            font weight, by default 'regular'
        bg : int | color, optional
            background brightness, by default `255` = white.
            can also be a color specification like `[128, 128, 128]`.

        """
        img = _get_text_image(text, size=size, family=family, weight=weight, bg=255)

        pal = [col, (bg * np.ones(3)).astype(np.uint8)]

        self.add_logo(img, pos=pos, width=width, depth=depth, plane=plane,
                      height=height, flip_x=flip_x, flip_y=flip_y, pal=pal, threshold=0.5)

    def save(self, i, path='.'):
        """write image layer i to `path`

        Parameters
        ----------
        i : int
            z-index of the layer
        path : str, optional
            path to store image. Filename will be like `slice_0001.png`, by default '.'
        """
        path = Path(path)
        imageio.imwrite(path / f'slice_{i:04d}.png', np.uint8(self.imgs[:, ::-1, i, :].transpose(1, 0, 2)))

    def save_images(self, path, i0=None, i1=None):
        """save all images between `i0` and `i1` to the directory `path`.

        Parameters
        ----------
        path : str | pathlib.Path
            directory into which to write the images
        i0 : int, optional
            initial index, by default 0
        i1 : int, optional
            final index, by default nz
        """

        # make output directory available & clean
        path = Path(path)
        if not path.is_dir():
            path.mkdir()
        else:
            files = list(path.glob('slice*.png'))
            if len(files) > 0:
                print('directory exists, deleting old files')
                for file in files:
                    file.unlink()

        # determine which slices to print
        i0 = i0 or 0
        i1 = i1 or self.nz - 1

        if (i0 != 0) or (i1 != self.nz - 1):
            print(f'only printing from index {i0} to {i1} = {(i1 - i0) * 2.54 / self.dpi_z:.2f} cm of it')

        indices = np.arange(i0, i1 + 1)

        list(tqdm(
            map(self.save, indices, repeat(path)),
            total=i1 + 1 - i0))
