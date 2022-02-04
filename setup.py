"""
Setup file for package `volrender`.
"""
import setuptools  # noqa
import sys
import warnings

try:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup
except ImportError:
    print('Error: numpy needs to be installed first')
    sys.exit(1)
import pathlib

PACKAGENAME = 'volrender'

extensions = Extension(name=f'{PACKAGENAME}._fortran', sources=[f'{PACKAGENAME}/fortran.f90'])

# the directory where this setup.py resides

HERE = pathlib.Path(__file__).absolute().parent

# function to parse the version from


def read_version():
    with (HERE / PACKAGENAME / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":

    def run_setup(extensions):
        setup(
            name=PACKAGENAME,
            description='simple volume rendering',
            version=read_version(),
            long_description=(HERE / "README.md").read_text(),
            long_description_content_type='text/markdown',
            url='https://github.com/birnstiel/astro3dprinting',
            author='Til Birnstiel',
            author_email='til.birnstiel@lmu.de',
            license='GPLv3',
            packages=setuptools.find_packages(),
            package_data={PACKAGENAME: [
                'volrender/fortran.f90',
            ]},
            include_package_data=True,
            ext_modules=[extensions],
            install_requires=[
                'matplotlib',
                'numpy'],
            python_requires='>=3.6',
            zip_safe=False,
            entry_points={
                'console_scripts': [
                    'volrender=volrender.volrender:main',
                    'render_movie=volrender.render_helper:main',
                    'image_stack=volrender.image_stack:main',
                ],
            }
        )

    try:
        run_setup(extensions)
    except Exception:
        warnings.warn('Setup with extensions did not work. Install fortran manually by issuing `make` in the diskwarp sub-folder')
        run_setup([])
