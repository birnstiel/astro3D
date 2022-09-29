__version__ = '0.0.1'

from pathlib import Path as _Path
from pkg_resources import resource_filename as _resource_filename

from ._fortran import fmodule
from . import lic

from . import volrender
from .volrender import TransferFunction
from .volrender import render_movie

__all__ = [
    'volrender',
    'fmodule',
    'lic',
    'Renderer',
    'TransferFunction',
    'render_movie',
]


def _get_path(fname=None, base='data'):
    """
    Helper function to retrieve file from packages directory.

    Argument
    --------

    fname : None | str
        if None: return base path
        if str: return path to file if it exists, else list files

    base : str
        base folder within package

    Output
    ------
    str : absolute path to file or base directory

    """
    p = _Path(_resource_filename(__name__, base))

    if (not p.exists()) and (base in ['output', 'data']):
        p.mkdir()

    if fname is None:
        return str(p)
    else:
        file = _Path(_resource_filename(__name__, str(_Path(base) / fname)))
        if file.exists() and (fname != ''):
            return str(file)
        else:
            print(f'Base path is \'{str(p)}\'')
            for file in p.glob('*'):
                print(f'- {str(file.name)}')
            return str(p)


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
    return _get_path(fname, base='output')


def get_data(fname=None):
    """Retrieve data from from packages DATA directory

    Argument
    --------

    fname : None | str
        if None: it returns the base path
        if string: returns file path if it exists, otherwise lists existing files

    Output
    ------
    str : absolute path to base directory or file

    """
    return _get_path(fname, base='data')
