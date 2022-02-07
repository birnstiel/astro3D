import numpy as np


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
