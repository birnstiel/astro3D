import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from matplotlib.widgets import Slider


class Renderer(object):

    def __init__(self, data, x=None, y=None, z=None, N=300, plot=True):

        if x is None:
            nx = data.shape[0]
            x = np.linspace(-nx / 2, nx / 2, nx)

        if y is None:
            ny = data.shape[1]
            y = np.linspace(-ny / 2, ny / 2, ny)

        if z is None:
            nz = data.shape[2]
            z = np.linspace(-nz / 2, nz / 2, nz)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x = x
        self.y = y
        self.z = z
        self.data = data

        self.phi = None
        self.theta = None

        self.N = N

        # set the observer grid onto which the data will be transformed
        c = np.linspace(-N / 2, N / 2, N)
        self.xo, self.yo, self.zo = np.meshgrid(c, c, c)

        # set up the transfer function

        self.transferfunction = TransferFunction()

        # set up the interpolation function

        self.f_interp = RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0.0)

        # the array holding the rendered image

        self.image = np.ones((N, N, 3))

        # if a plot should be done
        self.plot = plot
        if plot:
            self.f, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_aspect('equal')
            self.im = self.ax.imshow(self.image, origin='lower')

            pos = self.ax.get_position()
            self.slider_p_ax = self.f.add_axes([pos.x0, pos.y0 - 1 * pos.height / 20.0, pos.width, pos.height / 20.0])
            self.slider_t_ax = self.f.add_axes([pos.x0, pos.y0 - 2 * pos.height / 20.0, pos.width, pos.height / 20.0])

            self.slider_p = Slider(self.slider_p_ax, 'azimuth', 0.0, 360.0, valinit=0.0, valfmt='%.1f')
            self.slider_t = Slider(self.slider_t_ax, 'elevation', 0.0, 180.0, valinit=45.0, valfmt='%.1f')

            self.slider_p.on_changed(self.slider_update)
            self.slider_t.on_changed(self.slider_update)

            self.slider_update(None)

            plt.ion()
            plt.show()

    def slider_update(self, event):
        self.update(self.slider_t.val, self.slider_p.val)
        plt.draw()

    def rotate_data(self, phi, theta):
        """calculate the rotated data on the observer grid

        Parameters
        ----------
        phi : float
            azimuthal angle of the view in degree
        theta : float
            polar angle of the view in degree
        """
        phi, theta = np.deg2rad([phi, theta])

        qxR = self.xo * np.cos(phi) - self.yo * np.sin(phi) * np.cos(theta) + self.zo * np.sin(phi) * np.sin(theta)
        qyR = self.xo * np.sin(phi) + self.yo * np.cos(phi) * np.cos(theta) - self.zo * np.sin(theta) * np.cos(phi)
        qzR = self.yo * np.sin(theta) + self.zo * np.cos(theta)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

        # Interpolate onto Camera Grid
        self.data_obs = self.f_interp(qi).reshape((self.N, self.N, self.N))

    def render(self):
        """Do Volume Rendering"""
        self.image[...] = 0.0
        for dataslice in self.data_obs:
            r, g, b, a = self.transferfunction(dataslice)
            self.image[:, :, 0] = a * r + (1 - a) * self.image[:, :, 0]
            self.image[:, :, 1] = a * g + (1 - a) * self.image[:, :, 1]
            self.image[:, :, 2] = a * b + (1 - a) * self.image[:, :, 2]

        # np.clip(self.image, 0.0, 1.0, out=self.image)
        self.image = np.clip(self.image, 0.0, 1.0)

    def update(self, theta, phi):

        do_update = False

        # if the view changed, recalculate data
        if theta != self.theta or phi != self.phi:
            self.phi = phi
            self.theta = theta
            if self.plot:
                self.ax.set_title('interpolating ...')
                plt.draw()
            self.rotate_data(phi, theta)
            do_update = True

        # if other things change, manage them here and set do_update to True

        if do_update:

            if self.plot:
                self.ax.set_title('rendering ...')
                plt.draw()

            self.render()

            if self.plot:
                self.im.set_data(255 * self.image)
                self.ax.set_title('ready!')
                plt.draw()


class TransferFunction(object):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        x : array
            input data, 1D
        x0 : array | list
            positions of the gaussian peaks, shape = (N)
        A : array | list
            amplitude of each gaussian component, shape = (N)
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
        self.A = kwargs.pop('A', np.array([0.1, 0.1, 0.1]))
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
        A = np.expand_dims(self.A, axis=extra_dims)
        sigma = np.expand_dims(self.sigma, axis=extra_dims)
        colors = np.expand_dims(self.colors, axis=extra_dims)

        assert x0.shape == A.shape == sigma.shape, 'shapes of x0, A, and sigma must match'

        vals = colors[..., :, :] * A[..., :, None] * np.exp(-(x[..., None, None] - x0[..., :, None])**2 / (2 * sigma[..., :, None]**2))

        return vals.sum(-2).T


if __name__ == '__main__':

    print('reading data ... ', end='')
    with np.load('pluto_data.npz') as f:
        data = f['rho']
    print('Done!')

    vmax = data.max()
    datacube = LogNorm(vmin=vmax * 1e-4, vmax=vmax, clip=True)(data)

    render = Renderer(datacube)

    plt.show()
