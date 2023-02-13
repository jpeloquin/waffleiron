waffleiron is a Python package to create or modify [FEBio](https://github.com/febiosoftware/FEBio) models and analyze FEBio's simulation output.

Main features:

- Create FEBio models in Python.  The interface is designed to work at a higher level of abstraction than FEBio XML or [Gibbon](https://github.com/gibbonCode/GIBBON), closer to the level of abstraction in FEBio Studio (but with no GUI).  The aim is to keep user input as terse as possible through the use of sensible defaults and automatic logic.
- Import/export FEBio XML 3.0 and 2.5 to/from waffleiron model objects.
  - Good support for: `<Step>`, `<Globals>`, `<Mesh>`, `<MeshDomains>`, `<MeshData>`, `<Boundary>`, `<Rigid>`, `<Output>`.
  - Partial support for: `<Material>` (I've only implemented materials we use), `<Contact>` (tied-elastic works, the rest doesn't), `<LoadData>` (type="loadcurve" works; I haven't added type="math" yet).
  - No support for: `<Loads>`, `<Constraints>`, `<Discrete>`.
- Read data from xplt files into Python objects.  List metadata for variables, extract state data (except 'state/mesh state', which I haven't figured out yet), and extract geometry data.
- Read data from an FEBio text data file into a [pandas](https://github.com/pandas-dev/pandas) data frame (table).
- For a few materials, recalculate output variables independent of FEBio.  This has been used to calculate strain and stress for submaterials, especially fibers, in a mixture or multigeneration material.  This has helped identify a few FEBio bugs related to material orientation, which have been fixed upstream.

The main weaknesses of waffleiron are:

- It usually takes me a while to catch up to FEBio file format changes
- Test coverage is focused on the FEBio features used by me and my collaborators.

# Project status

Waffleiron was released on 2022-01-20 to https://github.com/jpeloquin/waffleiron.
I am working on user documentation.
If you are interested in testing it out, please consider [raising an issue](https://github.com/jpeloquin/waffleiron/issues) to describe what you're trying to do.
Development is guided by practical need.

# Using waffleiron

## Terms of use

waffleiron is licensed under the [AGPLv3](LICENSE).
You should read the entire license, but an informal explanation is given here for your convenience.
The license *allows* you to copy, modify, and use waffleiron, including for commercial use.
By doing so, you *incur obligations* when you redistribute waffleiron to retain the original copyright and license notifications, state your modifications, and provide your modified version of the source code to your users, even to those users who access waffleiron remotely or over the internet.

## Getting started

Waffleiron depends on:

- A working Python ≥ 3.8 environment.  Its dependencies are:
  - Required dependencies: numpy, scipy, lxml, pandas, matplotlib, psutil, shapely
  - Optional dependencies: mayavi (for tvtk; this can be a pain to install)
  - To run the test suite: pytest
- A working FEBio installation.  By default, waffleiron uses the command `febio` to start an FEBio process, but if the environment variable `FEBIO_CMD` is defined its value will be used instead.

TODO: Add minimal example.

## Support

Please feel free to [raise an issue](https://github.com/jpeloquin/waffleiron/issues) to ask for assistance and I will try to help.
Because waffleiron has been released into the wild very recently there are likely many difficulties to smooth over and I would like to hear about any you encounter.
FEBio also moves quickly and I may not be aware of breaking changes on FEBio's end.

However, responsibility for the accuracy or usefulness of any results you produce lies with you.
If you want me to verify that you are using the software in the correct manner and it is operating correctly, please contact me through the [Delaware Center for Musculoskeletal Research](https://sites.udel.edu/engr-dcmr/) to arrange contract work or collaboration.

# Contributing

Contributions to the codebase require a contributor license agreement (CLA).
The main motivation is to allow waffleiron to be re-licensed for use cases that were not originally anticipated.
This decision was made with the assumption that vast majority of contributions will continue to be made by the original author.
Please raise an issue or contact me through [DCMR](https://sites.udel.edu/engr-dcmr/) (DCMR) if you wish to contribute code.

To contribute documentation or other non-code resources to the waffleiron repo, please release it as [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

# Similar packages

Software that is similar in intent to waffleiron includes the following:

- [Gibbon](https://github.com/gibbonCode/GIBBON).  Requires Matlab.  Gibbon includes features for image analysis, which waffleiron never will, and has a much stronger set of features for meshing.  However, waffleiron has (in my opinion) more advanced capabilities for working with FEBio models and data.
- [pyFEBio](https://github.com/siboles/pyFEBio)
- [FEBio-Python](https://github.com/Nobregaigor/FEBio-Python)
- [interFEBio](https://github.com/andresutrera/interFEBio)
- [FEBio-Python-Pre-Post-processor](https://github.com/Nobregaigor/FEBio-Python-Pre-Post-processor)
- Unreleased software used in [Michele Tonutti's PhD thesis](http://dx.doi.org/10.13140/RG.2.2.34863.64165)

Waffleiron started development in February 2013 and Gibbon started in August 2014.
The development histories of the other packages are comparatively brief.
