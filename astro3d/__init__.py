from pathlib import Path as _Path
from importlib import metadata as _md

from ._fortran import fmodule
from . import image_stack
from . import memory_helper
from . import datasets

__version__ = _md.version('astro3d')


__all__ = [
    'image_stack',
    'fmodule',
    'memory_helper',
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
    """Retrieve data from from packages OUTPUT directory

    Argument
    --------

    fname : None | str
        if None: it returns the base path
        if string: returns file path if it exists, otherwise lists existing files

    Output
    ------
    str : absolute path to base directory or file

    """
    p = _PACKAGE_DIR / 'output'
    p.mkdir(exist_ok=True)

    if fname is None:
        return str(p)

    file = p / fname
    if file.exists() and (fname != ''):
        return str(file)

    print(f'Base path is \'{str(p)}\'')
    for file in p.glob('*'):
        print(f'- {str(file.name)}')
    return str(p)
