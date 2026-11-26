py_subdir = 'astro3d'

py3.install_sources([
  '__init__.py',
  'image_stack.py',
  'datasets.py',
  ], subdir: py_subdir)

# logo and example images used by the notebooks
install_subdir('figures', install_dir: py3.get_install_dir() / py_subdir)

fortran_module_name = '_fortran'

# OpenMP is optional: if found, meson adds the right compile (-fopenmp)
# and link flags through the dependency object; otherwise the !$OMP
# directives are ignored and the code runs serially.
openmp_dep = dependency('openmp', language: 'fortran', required: false)

# f2py only *generates* the C wrapper here; meson does the compiling.
# For code inside a Fortran module, f2py writes `<name>-f2pywrappers2.f90`.
fortran_cwrapper = custom_target(fortran_module_name,
  input: ['fortran.f90'],
  output: [fortran_module_name + 'module.c', fortran_module_name + '-f2pywrappers2.f90'],
  command: [py3, '-m', 'numpy.f2py', '--lower', '-m', fortran_module_name,
  '@INPUT@', '--build-dir', py_subdir]
)
py3.extension_module(fortran_module_name,
  'fortran.f90',
  fortran_cwrapper,
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies: [py_dep, openmp_dep],
  install: true,
  subdir: py_subdir
)
