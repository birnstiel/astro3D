import pytest

import astro3d


def test_version():
    assert isinstance(astro3d.__version__, str)
    assert astro3d.__version__.count('.') >= 1


def test_get_figure():
    logo = astro3d.get_figure('Logo_Origins2.png')
    assert logo.is_file()
    assert astro3d.get_figure('turbulent_box.mp4').is_file()


def test_get_figure_missing():
    with pytest.raises(FileNotFoundError, match='available are'):
        astro3d.get_figure('does_not_exist.png')
