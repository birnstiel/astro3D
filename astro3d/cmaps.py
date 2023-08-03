import base64
import io
from matplotlib.colors import _REPR_PNG_SIZE
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def RGB_cmap(name='magma', ncol=None, alpha=None, RGBA=False):
    """Create a copy of a default colormap with transparency.

    Parameters
    ----------
    name : str, optional
        colormap name, by default 'magma'
    ncol : int, optional
        number of color steps, defaults to 256
    alpha : None | array, optional
        can give an alpha array (0.0 ... 1.0), by default
        it will use a logistic function:
        1 - 1 / (exp((x - 0.25) / 0.1)**2 + 1)
    RGBA : bool, optional
        False (default): will turn transparency to white RGB
        True: will use actual transparency in RGBA

    Returns
    -------
    colormap
    """

    cmap = plt.get_cmap(name)

    # get a float RGBA color array from it of length `ncol`
    if alpha is None:
        ncol = ncol or 256
    else:
        ncol = ncol or len(alpha)
        if ncol != len(alpha):
            raise ValueError('ncol must be length of alpha')

    x = np.linspace(0, 1, ncol)
    rgba = (cmap(x) * 255).astype(np.uint8) / 255

    # define a transparency
    if alpha is None:
        alpha = 1 - 1 / (np.exp((x - 0.25)/0.1)**2 + 1)

    if RGBA:
        rgba[:, -1] = alpha
    else:
        rgba = rgba * alpha[:, None] + (1 - alpha[:, None])
        rgba[:, -1] = 1.0

    # make a cmap out of it
    return ListedColormap(rgba, f'white-{name}')


def get_cmyk_cmap(cmap, ncol=50):
    # first handle the input to make it a cmap from a string, an array or a list
    if isinstance(cmap, list):
        cmap = np.array(cmap)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if isinstance(cmap, np.ndarray):
        if issubclass(cmap.dtype.type, np.integer) or cmap.max() > 1:
            cmap = cmap / 255
        cmap = ListedColormap(cmap, 'input')

    # get a FLOAT RGBA color array from it of length `ncol`
    x = np.linspace(0, 1, ncol)
    rgba = cmap(x)

    # turn the alpha value into a white-background
    alpha = rgba[:, -1][:, None]
    rgba = rgba * alpha + (1 - alpha)
    rgba[:, -1] = 1.0

    # turn it into CMY with zero K
    cmyk = 1 - rgba
    cmyk[:, -1] = 0.0

    # turn this into a color map that returns CMYK values
    cmy_cmap = CMYListedColormap(cmyk, 'CMY')

    return cmy_cmap


class CMYListedColormap(ListedColormap):
    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]),
                    (_REPR_PNG_SIZE[1], 1))
        pixels = self(X, bytes=True)
        # begin change
        pixels = 255 - pixels
        pixels[:, :, -1] = 255
        # end change
        png_bytes = io.BytesIO()
        title = self.name + ' colormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')

        def color_block(color):
            # begin change
            color = 1 - color
            color[-1] = 1
            # end change
            hex_color = to_hex(color, keep_alpha=True)
            return (f'<div title="{hex_color}" '
                    'style="display: inline-block; '
                    'width: 1em; height: 1em; '
                    'margin: 0; '
                    'vertical-align: middle; '
                    'border: 1px solid #555; '
                    f'background-color: {hex_color};"></div>')

        return ('<div style="vertical-align: middle;">'
                f'<strong>{self.name}</strong> '
                '</div>'
                '<div class="cmap"><img '
                f'alt="{self.name} colormap" '
                f'title="{self.name}" '
                'style="border: 1px solid #555;" '
                f'src="data:image/png;base64,{png_base64}"></div>'
                '<div style="vertical-align: middle; '
                f'max-width: {_REPR_PNG_SIZE[0]+2}px; '
                'display: flex; justify-content: space-between;">'
                '<div style="float: left;">'
                f'{color_block(self.get_under())} under'
                '</div>'
                '<div style="margin: 0 auto; display: inline-block;">'
                f'bad {color_block(self.get_bad())}'
                '</div>'
                '<div style="float: right;">'
                f'over {color_block(self.get_over())}'
                '</div>')
