import hashlib

import pytest

from astro3d import datasets


def test_available():
    assert datasets.available() == ['mhddisk.npz', 'streamlines.npz', 'turbulentbox.npy']


@pytest.mark.parametrize('name', ['turbulentbox', 'turbulentbox.npy'])
def test_resolve_name(name):
    assert datasets._resolve(name) == 'turbulentbox.npy'


def test_unknown_name():
    with pytest.raises(ValueError, match='unknown data set'):
        datasets.fetch('nope')


def test_registry_hashes():
    for fname, known_hash in datasets.REGISTRY.items():
        algo, digest = known_hash.split(':')
        assert algo == 'sha256'
        assert len(digest) == 64, fname


def test_fetch_uses_cache_without_download(tmp_path, monkeypatch):
    """a file already in the cache with the right hash must not be downloaded"""
    content = b'some test data'
    (tmp_path / 'test.txt').write_bytes(content)
    monkeypatch.setattr(datasets._POOCH, 'path', tmp_path)
    monkeypatch.setitem(datasets._POOCH.registry, 'test.txt',
                        'sha256:' + hashlib.sha256(content).hexdigest())
    monkeypatch.setitem(datasets.REGISTRY, 'test.txt', datasets._POOCH.registry['test.txt'])

    # the base URL is not reachable in tests, so this only passes if pooch
    # takes the cached file
    assert datasets.fetch('test') == tmp_path / 'test.txt'


@pytest.mark.network
@pytest.mark.skipif('XXXXXXX' in datasets.BASE_URL, reason='data not yet on Zenodo')
@pytest.mark.parametrize('name', datasets.available())
def test_download(name):
    assert datasets.fetch(name).is_file()
