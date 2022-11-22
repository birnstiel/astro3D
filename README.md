[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/birnstiel/astro3d/pages%20build%20and%20deployment?label=docs&branch=docs)](https://birnstiel.github.io/astro3D/)


# 3D Printing Astrophysics Data

<span style="color:red; font-weight:bold; font-size:large">Work in progress.</span>

This package contains data, code, and notebooks to generate image stacks that can be 3D printed on Polyjet printers. See the [documentation](https://birnstiel.github.io/astro3D) for details.

![](Figure_2.jpg)

## Volume rendering

The package also includes some ability for volume rendering and line integral convolution plots for visualization.

To use the volume rendering, try out this:

    volrender turbulentbox.npy

or this:

    volrender -f rho pluto_data.npz

This should produce a volume rendering of the given data like this:

![](Figure_1.png)