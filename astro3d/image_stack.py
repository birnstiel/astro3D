from pathlib import Path
from itertools import repeat, cycle
import imageio
from colorsys import rgb_to_hsv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm, Normalize, to_rgb, LinearSegmentedColormap
from scipy.interpolate import interpn
from tqdm.auto import tqdm

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
VeroT_sRGB = np.array([255, 255, 255]) / 255
VeroC_sRGB = np.array([29, 85, 111]) / 255
VeroM_sRGB = np.array([149, 39, 87]) / 255
VeroY_sRGB = np.array([192, 183, 52]) / 255

# define the RGB values of CMY
C_sRGB = cmyk_to_rgb(np.array([100, 0, 0, 0])) / 255
M_sRGB = cmyk_to_rgb(np.array([0, 100, 0, 0])) / 255
Y_sRGB = cmyk_to_rgb(np.array([0, 0, 100, 0])) / 255

# colors defined in the VoxelPrinting Guide
BaseCyan = np.array([0, 89, 158]) / 255
BaseMagenta = np.array([161, 35, 99]) / 255
BaseYellow = np.array([213, 178, 0]) / 255
BaseBlack = np.array([30, 30, 30]) / 255
BaseWhite = np.array([220, 222, 216]) / 255


def makeslice(iz, z2, f_interp, coords, norm, path,
              levels=None, sigmas=None, fill=1.0,
              colors=None, f=None, bg=1.0, clip=[3.0, 3.0, 3.0],
              streamlines=None, radius=None):
    """
    Prints out one slice of the data (index `iz` within grid `z2`) and stores it as
    a file `slice_{iz:04d}.png` in the folder `path`.

    Several regions in density-space can be selected, each is defined by its entry in:
    - `levels` = the position in the normalized density (=when `norm` is applied to the
       density) where this color is applied
    - `sigmas` = the width of the gaussian region in normalized density
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

    path : str | Path
        the path into which to store the images

    levels : array
        array of `N_colors` float entries that define the density values of each color

    sigmas : array
        array of `N_colors` floats that describe the density width

    fill : float | array
        float or array of  `N_colors` floats that describe the filling factor of each color

    clip : array
        after how many sigmas to clip the color density

    colors : TBD
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

    path = Path(path)

    # update coordinates - only last entry changes
    _x, _y, _z = coords
    coords2 = np.array(np.meshgrid(_x, _y, z2[iz])).squeeze().reshape(3, -1).T

    # interpolate: note that we transpose as this is how the image will be safed
    new_layer = f_interp(coords2).reshape(len(_y), len(_x))

    # normalize data
    layer_norm = np.array(norm(new_layer))

    if levels is not None:
        # compute the different density contours (but exclude 0.0 which should never be colored)
        dist_sq = (np.array(levels)[None, None, :] - layer_norm[..., None])**2 / (2 * sigmas**2)
        dist_sq[layer_norm == 0.0] = np.inf
        dist_sq[dist_sq > np.array(clip)**2] = np.inf
        color_density = 1. / (1 + dist_sq) * fill
    else:
        # we are not using the levels, but the density directly
        color_density = layer_norm

    # create the dithered images
    layer_dithered = fmodule.dither_colors(color_density * fill)

    # --- begin adding streamlines ---
    if streamlines is not None:
        radius = radius or 2.5 * np.diff(_x)[0]
        mask = np.zeros([len(_y), len(_x)], dtype=bool)

        for line in streamlines:
            mask = mask | fmodule.mark_streamline(_x, _y, z2[iz], radius, line).T

        # clear other materials where the mask is true
        # and attach the streamlines as new material
        layer_dithered = np.where(mask[:, :, None], 0.0, layer_dithered)
        layer_dithered = np.concatenate((layer_dithered, mask[:, :, None]), axis=-1)
    # --- end streamlines ---

    # handle colors and f
    n_level = layer_dithered.shape[-1]
    if colors is None:
        colors = np.linspace(0.0, 1.0, n_level)[:, None] * np.ones(3)[None, :]
    if f is None:
        f = np.ones(n_level)

    # now replace the colors
    im = []

    old_colors = np.eye(n_level)

    for col_o, col_n, _f in zip(old_colors, colors, f):
        im += [color_replace(layer_dithered, col_o, col_n, f=_f)]

    # replace the background
    bg = bg * np.ones(3)
    im += [color_replace(layer_dithered, np.zeros(n_level), bg)]

    im = np.array(im).sum(0)

    # save as png
    imageio.imwrite(path / f'slice_{iz:04d}.png', np.uint8(255 * im))

    return layer_dithered


def process(data, height=10, dpi_x=600, dpi_y=600, dpi_z=1200, output_dir='slices',
            norm=None, pool=None, vmin=None, vmax=None, iz=None):
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

        vmax = vmax or 10**np.ceil(np.log10(data.max()))
        vmin = vmin or 1e-2 * vmax
        print('from {vmin:.2g} to {vmax:.2g}')
        norm = Norm(vmin=vmin, vmax=vmax, clip=True)

    # create original grid and interpolation function

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])

    def f_interp(coords):
        return fmodule.interpolate(x, y, z, data, coords)

    # calculate new grids

    n_z = int(dpi_z * height / 2.54)
    n_x = int(n_z * len(x) / len(z) / dpi_z * dpi_x)
    n_y = int(n_z * len(y) / len(z) / dpi_z * dpi_y)

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


def check_colors(im):
    colors = np.unique(im.reshape(-1, im.shape[-1]), axis=0)

    # we sort them "alphabetically", so white should be on the bottom
    colors = colors[np.lexsort(np.fliplr(colors).T)]

    return colors


def color_replace(im, orig_color, repl_col, f=[1], inplace=False):
    """replaces `orig_color` with `repl_col`.

    If a list of colors and a list of `f`s are given, then `orig_color``
    is replaced with that mix of colors.

    if `inplace` is `True`, then we are replacing colors in an image of matching shape
    and could leave the rest of the image as it is. This could replace a single
    color in the image with another color.

    if `inplace` is false, the shape does not need to match, for example if we want to construct an
    image from more or less than 3 layers. For 2 layers, the input information is `(nx, ny, 2)` but
    the image should be `(nx, ny, 3)`, for example.
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
        im_repl = np.where(color_mask[:, :, None], repl_cols[0, None, None], im)
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
    bins = np.linspace(0, 1, 100)
    counts, _ = np.histogram(np.array(norm(data.ravel())), bins=bins)

    if colors is not None:
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
    if (levels is not None) and (sigmas is not None) and (clips is not None):
        if colors is None:
            # get default colors
            colors = np.array([to_rgb(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(levels)]])
        for i, (_level, _sig, _clip) in enumerate(zip(levels, sigmas, clips)):
            ax.axvline(_level, ls='--', c=mix[i])

            _y = 10**(np.mean(np.log10(ax.get_ylim())) * (1 + 0.1 * i))

            ax.errorbar(_level, _y,
                        xerr=[
                            [_sig],
                            [_sig]], c=mix[i], capsize=5)
            ax.errorbar(_level, _y,
                        xerr=[
                            [_clip * _sig],
                            [_clip * _sig]], c=mix[i], capsize=5, alpha=0.5)

        # estimate the filling factor
        dn = np.array(norm(data.ravel())).reshape(data.shape)
        dist_sq = (np.array(levels)[None, None, :] - dn[..., None])**2 / (2 * sigmas**2)
        dist_sq[dn == 0.0] = np.inf
        dist_sq[dist_sq > np.array(clips)**2] = np.inf
        color_density = 1. / (1 + dist_sq) * fill
        ff = color_density.sum() / np.product(data.shape)
    else:
        ff = (norm(data.ravel())).sum() / np.product(data.shape)

    ticks = np.arange(*np.round(np.log10(np.array(norm.inverse([0, 1])))) + [0, 1])

    ax2 = ax.secondary_xaxis('top', functions=(norm.inverse, norm))
    ax2.set_xlabel('original density')
    ax2.get_xaxis().set_major_locator(ticker.LogLocator())
    ax2.get_xaxis().set_ticks(10.**ticks)

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


class IStack(object):
    def __init__(self, directory, dpi_x, dpi_y, dz):
        self.directory = Path(directory)
        self.files = sorted(list(self.directory.glob('*.png')))

        # read image stack - the transpose makes the order x, y, z
        collection = imread_collection(str(self.directory / '*.png'))
        self.imgs = collection.concatenate()
        self.imgs = self.imgs.transpose(2, 1, 0, 3)

        # get image sizes
        self.nx, self.ny = self.imgs.shape[:2]
        self.nz = len(self.files)
        self._x = np.arange(self.nx)
        self._y = np.arange(self.ny)

        # set printer properties
        self.dx = 2.54 / dpi_x
        self.dy = 2.54 / dpi_y
        self.dz = dz

        self.dpi_x = dpi_x
        self.dpi_y = dpi_y
        self.dpi_z = 2.54 / dz

        # determine the palette
        self._colors = None
        self._ncol = None
        self._counts = None

        # which indices are empty
        self._empty_indices = None

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

    def _get_colors(self, N=10):
        """Get all colors present in a random sub-sample of N slices.
        """
        print(f'getting colors from {N} sample images ... ', flush=True, end='')
        idx = np.random.randint(0, high=self.nz - 1, size=N)
        self._colors = check_colors(self.imgs[:, :, idx, :])
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
        new_col = np.array(new_col)
        mask = (self.imgs == self.colors[i_col][None, None, None, :]).all(-1)
        self.imgs = np.where(mask[:, :, :, None], new_col[None, None, None, :], self.imgs)
        self.colors[i_col, :] = new_col

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
            cc = ax[1, i_column].imshow(self.counts[:, :, ic], vmin=0, vmax=vmax, origin='lower', cmap=cmap)
            ax[1, i_column].set_aspect(self.dpi_y / self.dpi_x)
            f.colorbar(cc, cax=ax[0, i_column], orientation='horizontal')

        return f, ax

    def show_info(self, empty_indices=None):
        "prints some infos about the image stack (size, filling factor, ...)"
        # indices of the non-transparent
        empty_indices = empty_indices or self.empty_indices
        rest = self._get_nonempty(empty_indices)

        print(f'There are {self.ncol} colors in this image:')
        for i, row in enumerate(self.colors):
            print(f'- {list(row)}' + (i in empty_indices) * ' (transp.)')

        print(f'{self.nz} files')
        print(f'dimension = {self.nx * self.dx:.2f} x {self.ny * self.dy:.2f} x {self.nz * self.dz:.2f} cm')
        print(f'filling fraction: {1 - self.counts[:, :, empty_indices].sum() / (self.nx * self.ny * self.nz):.2%}')

        print(f'nr of fully transparent columns: {(self.counts[:, :, rest].sum(-1) == 0).sum() / (self.nx * self.ny * self.nz):.2%}')
        print(f'most opaque pixel has {self.counts[:, :, rest].sum(-1).max():n} filled pixels (={self.counts[:, :, rest].sum(-1).max() / self.nz:.2%} of all layers are filled)')

        print('mean counts in non-transparent columns: ' + ', '.join([f'{np.mean(self.counts[:, :, i][self.counts[:,:,i]!=0]):.2g}' for i in range(self.ncol)]))

    def show_transparency_estimate(self, empty_indices=None):
        "shows a map of the estimated projected transparency"

        empty_indices = empty_indices or self.empty_indices
        rest = self._get_nonempty(empty_indices)

        f, axs = plt.subplots(1, 3, figsize=(11, 3), dpi=100, tight_layout=True)

        opts = dict(cmap='gray', origin='upper')
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

    def get_top_view(self, empty_indices=None, n_tauone=7, bg=[0, 0, 0], view='xy', backward=False):
        """computes an approximate render view from top, assuming the optical depth is around `n_tauone` pixels.

        Parameters
        ----------
        empty_indices : array, optional
            indices of the colors in self.colors that should be treated as transparent,
            by default it uses the class property

        n_tauone : int, optional
            after how many pixels the print becomes optically thick, by default 7

        bg : array, optional
            background image color, default black, i.e. [0, 0, 0]

        view : str
            viewing direction: 'xy' (default), 'xz', 'yz'

        backward : bool
            view from opposite direction, default: False

        Returns
        -------
        array
            rendered image
        """

        if view == 'xy':
            transpose = [0, 1, 2, 3]
        elif view == 'xz':
            transpose = [0, 2, 1, 3]
        elif view == 'yz':
            transpose = [1, 2, 0, 3]
        else:
            raise ValueError('invalid view')

        # which colors in stack.colors are to be treated as transparent
        empty_indices = empty_indices or self.empty_indices

        # start with black background (white background would require substracting the inverse colors or so)
        image = np.ones_like(self.imgs.transpose(*transpose)[:, :, 0, :], dtype=float) * bg

        # this is the transfer function exp(-tau). We assume that about 7 pixels are tau=1
        exp = np.exp(-1. / n_tauone)
        trans_fct = exp * np.ones(image.shape[:2], dtype=image.dtype)

        indices = np.arange(self.imgs.transpose(*transpose).shape[2])
        if backward:
            indices = indices[::-1]

        for i in indices:
            slice = self.imgs.transpose(*transpose)[:, :, i, :]

            # this array is set to 0 for every transparent pixel, and 1 for the rest
            transparency_factor = np.ones_like(slice[:, :, 0], dtype=image.dtype)
            for i_t in empty_indices:
                transparency_factor[(slice == self.colors[i_t]).all(-1)] = 1 / exp

            _tf = trans_fct * transparency_factor

            image = image * _tf[:, :, None] + (1 - _tf[:, :, None]) * slice

        return image

    def show_top_view(self, ax=None, **kwargs):
        "plots the top view generated with `get_top_view` (to which kwargs are passed)"
        image = self.get_top_view(**kwargs)

        i, j = kwargs.get('view', 'xy')
        ratio = getattr(self, f'dpi_{j}') / getattr(self, f'dpi_{i}')

        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.figure

        ax.imshow(image / 255)
        ax.set_aspect(ratio)

        return f, ax

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
        imageio.imwrite(path / f'slice_{i:04d}.png', np.uint8(self.imgs[:, :, i, :].transpose(1, 0, 2)))

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
