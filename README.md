[![tests](https://github.com/birnstiel/astro3D/actions/workflows/tests.yml/badge.svg?branch=v2)](https://github.com/birnstiel/astro3D/actions/workflows/tests.yml)

# astro3d: 3D printing astrophysical data

`astro3d` turns 3D data sets (simulation boxes, point clouds, ...) into
stacks of dithered image slices for multi-material [PolyJet](https://en.wikipedia.org/wiki/Material_jetting)
printers. It also provides tools to inspect, modify, and preview the
resulting image stacks before printing.

<video src="astro3d/figures/turbulent_box.mp4" width="400" autoplay loop muted></video>

> **Note:** version 2 is under active development.

## Installation

The recommended way is [pixi](https://pixi.sh), which also provides the
Fortran compiler:

```bash
git clone https://github.com/birnstiel/astro3D.git
cd astro3D
pixi install       # creates the environment and builds astro3d
pixi run test      # optional: run the test suite
```

Alternatively, with `pip` (needs a Fortran compiler, e.g. `gfortran`, to
build the extension):

```bash
pip install git+https://github.com/birnstiel/astro3D.git@v2
```

## Quick start

```python
import numpy as np
import astro3d
from astro3d import image_stack

# example data: a 256^3 turbulent box (downloaded on first use)
data = np.load(astro3d.datasets.fetch('turbulentbox'))

# slice it into PNG images for a 5 cm cube; this writes ~1850 images,
# use e.g. `iz=[0, 900]` to only compute a few layers for a preview
output = astro3d.get_output('turbulent_box')
image_stack.process(data, height=5, output_dir=output)

# load the image stack and preview how the print will look from three sides
stack = image_stack.IStack(output)
stack.three_views()
```

## Documentation

The [notebooks](notebooks/) show complete examples:

| notebook | content |
|---|---|
| `01-01_slices_turbulent_box` | slicing a density field into a printable image stack |
| `01-02_analyze_turbulent_box` | inspecting and modifying an image stack, adding a logo |
| `02-01_slices_MHD` | multi-color contours and magnetic field lines |
| `02-02_analyze_MHD` | inspecting the MHD image stack |

## Data

Example data sets are downloaded and cached automatically, see
`astro3d.datasets.available()`. Set `ASTRO3D_DATA_DIR` to use a local
folder instead. Generated output goes into `./output` (or
`ASTRO3D_OUTPUT_DIR`).

## License

GPL v3, see [LICENSE](LICENSE).
