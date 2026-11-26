import pytest

import astro3d


def test_version():
    assert isinstance(astro3d.__version__, str)
    assert astro3d.__version__.count('.') >= 1


def test_get_figure():
    logo = astro3d.get_figure('Logo_Origins2.png')
    assert logo.is_file()
    assert astro3d.get_figure('turbulent_box.mp4').is_file()


def test_get_output_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ASTRO3D_OUTPUT_DIR', raising=False)
    assert astro3d.get_output() == tmp_path / 'output'
    assert (tmp_path / 'output').is_dir()
    assert astro3d.get_output('slices') == tmp_path / 'output' / 'slices'


def test_get_output_env(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTRO3D_OUTPUT_DIR', str(tmp_path / 'elsewhere'))
    assert astro3d.get_output('x') == tmp_path / 'elsewhere' / 'x'


def test_get_figure_missing():
    with pytest.raises(FileNotFoundError, match='available are'):
        astro3d.get_figure('does_not_exist.png')
