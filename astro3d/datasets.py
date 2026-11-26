"""
Example data sets used in the documentation notebooks.

The files are downloaded on first use and cached locally (via `pooch`),
the download is verified with a SHA256 hash.

    >>> import numpy as np
    >>> import astro3d
    >>> data = np.load(astro3d.datasets.fetch('turbulentbox'))

By default, the cache is the OS-specific cache folder (e.g.
`~/Library/Caches/astro3d` on macOS, `~/.cache/astro3d` on Linux).
Set the environment variable `ASTRO3D_DATA_DIR` to use another folder,
for example one where you already have the files: files in there that
match the hash are used without downloading.
"""
from pathlib import Path

import pooch

__all__ = ['fetch', 'available', 'data_dir']

# Zenodo record holding the files.
# TODO: replace with the real DOI once the data is uploaded to Zenodo,
#       e.g. "doi:10.5281/zenodo.1234567/"
BASE_URL = 'doi:10.5281/zenodo.XXXXXXX/'

# file name -> hash. If a file changes, upload it as a new Zenodo version
# and update the DOI and hash here.
REGISTRY = {
    # 256^3 turbulent box (density), float64
    'turbulentbox.npy': 'sha256:1f29fbeebcbb8c2b451591036849259721a5642c52777a98278941c1e6646faa',
    # MHD disk density, key `arr_0`
    'mhddisk.npz': 'sha256:73e84b17a170ce3cf96d7381006d8921874b392c7ae032c9cc2405049dd005e8',
    # magnetic field lines of the MHD disk, key `pathes` (needs allow_pickle=True)
    'streamlines.npz': 'sha256:734728be671c9bb80ad02a277eb493c359a6e249a8bf9ae22388986ed91e833b',
}

_POOCH = pooch.create(
    path=pooch.os_cache('astro3d'),
    base_url=BASE_URL,
    registry=REGISTRY,
    env='ASTRO3D_DATA_DIR',
)


def available():
    """List the names of all available data sets."""
    return sorted(REGISTRY)


def data_dir():
    """Return the folder where data sets are cached."""
    return Path(_POOCH.abspath)


def _resolve(name):
    """Allow names with or without file extension, e.g. 'turbulentbox'."""
    if name in REGISTRY:
        return name
    matches = [f for f in REGISTRY if Path(f).stem == name]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f'unknown data set {name!r}, available are: {", ".join(available())}')


def fetch(name):
    """Return the local path of a data set, downloading it if needed.

    Parameters
    ----------
    name : str
        name of the data set, with or without extension,
        see `astro3d.datasets.available()`

    Returns
    -------
    pathlib.Path
        path to the local (verified) file
    """
    return Path(_POOCH.fetch(_resolve(name), progressbar=True))
