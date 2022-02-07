import argparse
from turtle import update
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.interpolate import RegularGridInterpolator
from matplotlib.widgets import Slider
import warnings
try:
    from volrender import fmodule
    fmodule_available = True
except Exception:
    warnings.warn('Fortran routine unavailable, code will be slightly slower')
    fmodule_available = False


class Renderer(object):

    def __init__(self, data, N=300, interactive=False, diagnostics=False, tf=None):

        nx = data.shape[0]
        x = np.linspace(-nx / 2, nx / 2, nx)

        ny = data.shape[1]
        y = np.linspace(-ny / 2, ny / 2, ny)

        nz = data.shape[2]
        z = np.linspace(-nz / 2, nz / 2, nz)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        self.data_obs = np.zeros([N, N, N])

        self._f_int = fmodule_available

        self.phi = 0.0
        self.theta = 0.0

        # set the observer grid onto which the data will be transformed
        self._N = N
        nmax = max(nx / 2, ny / 2, nz / 2)
        cx = np.linspace(-nmax, nmax, N)
        cy = np.linspace(-nmax, nmax, N)
        cz = np.linspace(-nmax, nmax, N)
        self.xo, self.yo, self.zo = np.meshgrid(cx, cy, cz, indexing='ij')

        # set up the transfer function

        if tf is None:
            tf = TransferFunction()
        self.transferfunction = tf

        # set up the interpolation function
        if not self._f_int:
            self.init_interpolation()

        # the array holding the rendered image

        self.image = np.ones((N, N, 3))

        # if a plot should be done
        self.interactive = interactive
        self.diagnostics = diagnostics
        if interactive:
            # make the figure; make it wider if the histogram is plotted
            self.f = plt.figure(figsize=(6 * (1 + diagnostics), 10))
            ratio = self.f.get_figwidth() / self.f.get_figheight()
            # this is adding a square axis in the top ~half
            # will cover only the left side if the histogram is added
            self.ax = self.f.add_axes([
                0.1,
                1 - 0.9 * ratio / (1 + diagnostics),
                0.8 / (1 + diagnostics),
                0.8 * ratio / (1 + diagnostics)])
            self.ax.set_aspect('equal')
            self.im = self.ax.imshow(self.image.transpose(1, 0, 2), origin='lower')

            pos = self.ax.get_position()

            # if the histogram should be shown
            if diagnostics:
                # add another axis
                self._axd = self.ax = self.f.add_axes([
                    0.55,
                    1 - 0.9 * ratio / (1 + diagnostics),
                    0.8 / (1 + diagnostics),
                    0.8 * ratio / (1 + diagnostics)])

                counts, edges = np.histogram(self.data.ravel(), 200, density=True)
                centers = 0.5 * (edges[1:] + edges[:-1])
                self._centers = centers

                # display the denisty histogram
                self._axd.plot(centers, counts, '0.5', ds='steps')

                # display the colors
                rgba = self.transferfunction(centers)
                tf_image = (rgba[:3, :, None] * np.ones(100)[None, None, :]).T
                tf_image = Normalize()(tf_image)

                self._fillmax = counts.max()
                self._fillmin = self._fillmax * 1e-10

                self._im_diag = self._axd.imshow(tf_image, extent=[*centers[[0, -1]], self._fillmin, self._fillmax],
                                                 interpolation='none', interpolation_stage='data', resample=True)
                self._fill = self._axd.fill_between(centers, rgba[-1], self._fillmax, fc='white', ec='k', zorder=0)
                self._axd.set_ylim(self._fillmin, self._fillmax)
                self._axd.set_aspect('auto')
                self._axd.set_yscale('log')

            # slider axes

            hw_ratio = 20.0

            self.slider_phi_ax = self.f.add_axes([pos.x0, pos.y0 - 2 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])
            self.slider_the_ax = self.f.add_axes([pos.x0, pos.y0 - 3 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])

            self.slider_v00_ax = self.f.add_axes([pos.x0, pos.y0 - 4 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])
            self.slider_v01_ax = self.f.add_axes([pos.x0, pos.y0 - 5 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])
            self.slider_v02_ax = self.f.add_axes([pos.x0, pos.y0 - 6 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])

            self.slider_sig_ax = self.f.add_axes([pos.x0, pos.y0 - 7 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])

            self.slider_al1_ax = self.f.add_axes([pos.x0, pos.y0 - 8 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])
            self.slider_al2_ax = self.f.add_axes([pos.x0, pos.y0 - 9 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])
            self.slider_al3_ax = self.f.add_axes([pos.x0, pos.y0 - 10 * pos.height / hw_ratio, pos.width, pos.height / hw_ratio])

            # sliders

            self.slider_phi = Slider(self.slider_phi_ax, 'azimuth', 0.0, 360.0, valinit=0.0, valfmt='%.1f')
            self.slider_the = Slider(self.slider_the_ax, 'elevation', 0.0, 180.0, valinit=0.0, valfmt='%.1f')

            self.slider_v00 = Slider(self.slider_v00_ax, '$v_0$', 0.0, 1.0, valinit=0.2, valfmt='%.1f')
            self.slider_v01 = Slider(self.slider_v01_ax, '$v_1$', 0.0, 1.0, valinit=0.4, valfmt='%.1f')
            self.slider_v02 = Slider(self.slider_v02_ax, '$v_2$', 0.0, 1.0, valinit=0.9, valfmt='%.1f')

            self.slider_sig = Slider(self.slider_sig_ax, '$sig$', -2, 2, valinit=np.log10(0.02), valfmt='%.1f')

            self.slider_al1 = Slider(self.slider_al1_ax, 'alpha$_1$', 0, 1, valinit=1e-2, valfmt='%.1f')
            self.slider_al2 = Slider(self.slider_al2_ax, 'alpha$_2$', 0, 1, valinit=5e-2, valfmt='%.1f')
            self.slider_al3 = Slider(self.slider_al3_ax, 'alpha$_3$', 0, 1, valinit=1e-1, valfmt='%.1f')

            # set update

            self.slider_phi.on_changed(self.slider_update)
            self.slider_the.on_changed(self.slider_update)

            self.slider_v00.on_changed(self.slider_update)
            self.slider_v01.on_changed(self.slider_update)
            self.slider_v02.on_changed(self.slider_update)

            self.slider_sig.on_changed(self.slider_update)

            self.slider_al1.on_changed(self.slider_update)
            self.slider_al2.on_changed(self.slider_update)
            self.slider_al3.on_changed(self.slider_update)

            self.update(self.slider_the.val, self.slider_phi.val, do_update=True)

            plt.show()

    def slider_update(self, event):
        self.transferfunction.x0[0] = self.slider_v00.val
        self.transferfunction.x0[1] = self.slider_v01.val
        self.transferfunction.x0[2] = self.slider_v02.val

        self.transferfunction.sigma = 10.**self.slider_sig.val * np.ones(3)

        self.transferfunction.colors[0, -1] = self.slider_al1.val
        self.transferfunction.colors[1, -1] = self.slider_al2.val
        self.transferfunction.colors[2, -1] = self.slider_al3.val

        self.update(self.slider_the.val, self.slider_phi.val)

        if self.diagnostics:
            self.update_diagnostics()
        plt.draw()

    def init_interpolation(self):
        self.f_interp = RegularGridInterpolator((self.x, self.y, self.z), self.data, bounds_error=False, fill_value=0.0)

    def update_diagnostics(self):
        rgba = self.transferfunction(self._centers)
        tf_image = (rgba[:3, :, None] * np.ones(100)[None, None, :]).T
        tf_image = Normalize()(tf_image)
        self._im_diag.set_data(Normalize()(tf_image))
        self._axd.collections.clear()
        self._fill = self._axd.fill_between(self._centers, rgba[-1], self._fillmax, fc='white', ec='k', zorder=0)

    def render(self, phi=None, theta=None, update=True):
        """render the data from azimuth `phi`, elevation `theta`. Store in self.image

        Parameters
        ----------
        phi : float
            azimuthal angle in degree
        theta : float
            polar angle in degree
        update : bool
            if false, only re-render, no need to interpolate, default=True
        """
        if update:
            if phi is None:
                phi = self.phi
            if theta is None:
                theta = self.theta

            phi, theta = np.deg2rad([phi, theta])

            qxR = self.xo * np.cos(phi) - self.yo * np.sin(phi) * np.cos(theta) + self.zo * np.sin(phi) * np.sin(theta)
            qyR = self.xo * np.sin(phi) + self.yo * np.cos(phi) * np.cos(theta) - self.zo * np.sin(theta) * np.cos(phi)
            qzR = self.yo * np.sin(theta) + self.zo * np.cos(theta)
            qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

            # Interpolate onto Camera Grid
            if self._f_int:
                self.data_obs = fmodule.interpolate(self.x, self.y, self.z, self.data, qi).reshape((self._N, self._N, self._N))
            else:
                self.data_obs = self.f_interp(qi).reshape((self._N, self._N, self._N))

        self.image = fmodule.render(self.data_obs,
                                    self.transferfunction.x0,
                                    self.transferfunction.sigma,
                                    self.transferfunction.colors)

    def update(self, theta, phi, do_update=False):

        # if the view changed, recalculate data
        if theta != self.theta or phi != self.phi:
            self.phi = phi
            self.theta = theta
            do_update = True

        # if other things change, manage them here and set do_update to True

        self.render(phi, theta, update=do_update)

        if self.interactive:
            self.im.set_data(Normalize()(self.image.transpose(1, 0, 2)))
            plt.draw()

    def plot(self, norm=None, diagnostics=False, L=None):
        """Make a plot of the rendered image

        Parameters
        ----------
        norm : norm, optional
             that was used to scale the data, assuming linear 0...1 if none is given, by default None
        diagnostics : bool, optional
            if true, plot also diagnostics of the transfer function and image, by default False
        L : float, optional
            box length to rescale things, by default None

        Returns
        -------
        figure, axes
        """

        if norm is None:
            print('no norm given assuming linear from 0 to 1')
            norm = Normalize()

        if L is None:
            L = self.data.shape[0]

        # make the plot
        if diagnostics:
            f, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=200, gridspec_kw={'wspace': 0.3})
            ax = axs[0]
        else:
            f, ax = plt.subplots(figsize=(4, 4), dpi=200)

        ax.imshow(Normalize()(self.image), extent=[-L / 2, L / 2, -L / 2, L / 2], rasterized=True)

        ax.set_xlabel('x [au]')
        ax.set_ylabel('y [au]')

        # make axis for the colorbar

        pos = ax.get_position()
        cax = f.add_axes([pos.x1 + pos.height / 20 / 5, pos.y0, pos.height / 20, pos.height])

        # make a color map based on the transfer function

        x = np.linspace(0, 1, 200)
        tf_image = self.transferfunction(x)
        tf_image = tf_image[:3, :].T
        tf_image = Normalize()(tf_image)
        col = ListedColormap(tf_image)

        # add a colorbar just based on the norm and colormap

        cb = f.colorbar(cm.ScalarMappable(norm=norm, cmap=col), cax=cax)
        cb.set_label('$\\rho$ [g cm$^{-3}$]')

        # make the plot

        # make the transfer function plot
        if diagnostics:
            ax = axs[1]
            ax.set_facecolor('k')

            counts, edges = np.histogram(self.data.ravel(), 200, density=True)
            centers = 0.5 * (edges[1:] + edges[:-1])

            rgba = self.transferfunction(centers)

            ax.plot(centers, counts, '0.5', ds='steps')

            tf_image = (rgba[:3, :, None] * np.ones(100)[None, None, :]).T
            tf_image = Normalize()(tf_image)

            ymax = counts.max()
            ymin = ymax * 1e-10

            ax.imshow(tf_image, extent=[*centers[[0, -1]], ymin, ymax],
                      interpolation='none', interpolation_stage='data', resample=True)
            ax.fill_between(centers, rgba[-1], ymax, fc='white', ec='k', zorder=0)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('auto')
            ax.set_yscale('log')

            return f, ax


class TransferFunction(object):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        x : array
            input data, 1D
        x0 : array | list
            positions of the gaussian peaks, shape = (N)
        sigma : array | list
            width of each gaussian component, shape = (N)
        colors : array | list
            4-element RGBA color for each gaussian component, shape = (N, 4)

        Returns
        -------
        array
            RGBA values for each element in x, shape = (len(x), 4)
        """
        self.x0 = kwargs.pop('x0', np.array([0.2, 0.4, 0.9]))
        self.sigma = kwargs.pop('sigma', [0.02, 0.02, 0.02])
        self.colors = kwargs.pop('colors',
                                 np.array([
                                     [1.0, 0.0, 0.0, 1e-2],
                                     [0.0, 1.0, 0.0, 5e-2],
                                     [0.0, 0.0, 1.0, 1e-1],
                                 ])
                                 )

    def __call__(self, x):
        """returns RGBA values for input array `x`

        Parameters
        ----------
        x : array
            input data, 1D

        Returns
        -------
        array
            RGBA values for each element in x, shape = (len(x), 4)
        """

        extra_dims = tuple(np.arange(x.ndim))

        x0 = np.expand_dims(self.x0, axis=extra_dims)
        sigma = np.expand_dims(self.sigma, axis=extra_dims)
        colors = np.expand_dims(self.colors, axis=extra_dims)

        assert x0.shape == sigma.shape == colors.shape[:2], 'shapes of x0, colors, and sigma must match (colors with one extra-dimension of len=4)'

        vals = colors[..., :, :] * np.exp(-(x[..., None, None] - x0[..., :, None])**2 / (2 * sigma[..., :, None]**2))

        return vals.sum(-2).T


def main():
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description='simple volume rendering example', formatter_class=RTHF)
    PARSER.add_argument('-f', '--field', help='if dictionary based npz file is used, use this field from the file', type=str, default=None)
    PARSER.add_argument('-d', '--diagnostics', help='show diagnostics in interactive view', action='store_true', default=False)
    PARSER.add_argument('filename', help='which .npz file to read', type=str)
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

    Renderer(datacube, interactive=True, diagnostics=ARGS.diagnostics)
    plt.show()


if __name__ == '__main__':
    main()
    plt.show()
