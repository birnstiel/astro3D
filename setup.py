"""
Setup file for package `astro3d`.
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

PACKAGENAME = 'astro3d'

extensions = [
    Extension(
        name=f'{PACKAGENAME}._fortran',
        sources=[f'{PACKAGENAME}/fortran.f90'],
        extra_f90_compile_args=["-fopenmp"],
        extra_link_args=["-lgomp"],
    ),
    Extension(
        name=f'{PACKAGENAME}._lic',
        sources=[f'{PACKAGENAME}/lic.f90'],
    ),
]

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

    def run_setup(extension):
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
                'astro3d/fortran.f90',
            ]},
            include_package_data=True,
            ext_modules=extensions,
            install_requires=['matplotlib', 'numpy', 'scikit-image', 'Pillow', 'scipy', 'tqdm'],
            python_requires='>=3.6',
            zip_safe=False,
            entry_points={
                'console_scripts': [
                    'volrender=astro3d.CLI:volrender_CLI',
                    'render_movie=astro3d.CLI:render_movie_CLI',
                    'image_stack=astro3d.CLI:image_stack_CLI',
                ],
            }
        )

    try:
        run_setup(extensions)
    except Exception:
        try:
            warnings.warn('OpenMP/gfortran not available, will try without')
            for extension in extensions:
                extension.extra_f90_compile_args = []
                extension.extra_link_args = []
            run_setup(extensions)
        except Exception:
            warnings.warn('Setup with extensions did not work. Install fortran manually by issuing `make` in the diskwarp sub-folder')
            run_setup([])
