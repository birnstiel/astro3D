import os as _os
from pathlib import Path as _Path
from importlib import metadata as _md

from ._fortran import fmodule
from . import image_stack
from . import datasets

__version__ = _md.version('astro3d')


__all__ = [
    'image_stack',
    'fmodule',
    'datasets',
    'get_figure',
    'get_output',
]

_PACKAGE_DIR = _Path(__file__).parent


def get_figure(fname):
    """Return the path to a figure shipped with the package (e.g. the logo).

    Parameters
    ----------
    fname : str
        file name within the package's `figures` folder

    Returns
    -------
    pathlib.Path
        absolute path to the file
    """
    path = _PACKAGE_DIR / 'figures' / fname
    if not path.is_file():
        files = ', '.join(sorted(p.name for p in path.parent.glob('*')))
        raise FileNotFoundError(f'no figure {fname!r}, available are: {files}')
    return path


def get_output(fname=None):
    """Return the folder for generated output (e.g. image stacks).

    This is the folder `output` in the current working directory, or the
    folder given by the environment variable `ASTRO3D_OUTPUT_DIR`. The base
    folder is created if needed, `fname` is not (the caller creates it).

    Parameters
    ----------
    fname : None | str
        if given, return the path of this file/folder within the output folder

    Returns
    -------
    pathlib.Path
        absolute path to the output folder or to `fname` within it
    """
    base = _Path(_os.environ.get('ASTRO3D_OUTPUT_DIR', 'output')).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base if fname is None else base / fname
