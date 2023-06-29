# SkyLLH

[![Tests](https://github.com/icecube/skyllh/actions/workflows/pythonpackage.yml/badge.svg)](#)
[![Docs](https://github.com/icecube/skyllh/actions/workflows/documentation.yml/badge.svg)](https://icecube.github.io/skyllh/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/skyllh)](https://pypi.org/project/skyllh/)

[[Full documentation]](https://icecube.github.io/skyllh/).

The SkyLLH framework is an open-source Python3-based package licensed under the GPLv3 license. It provides a modular framework for implementing custom likelihood functions and executing log-likelihood ratio hypothesis tests. The idea is to provide a class structure tied to the mathematical objects of the likelihood functions, rather than to entire abstract likelihood models.

# Installation

## Using pip

The latest `skyllh` release can be installed from [PyPI](https://pypi.org/project/skyllh/) repository:
```bash
pip install skyllh
```

The current development version can be installed using pip:
```bash
pip install git+https://github.com/icecube/skyllh.git#egg=skyllh
```

Optionally, the editable package version with a specified reference can be installed by:
```bash
pip install -e git+https://github.com/icecube/skyllh.git@[ref]#egg=skyllh
```
where
- `-e` is an editable flag
- `[ref]` is an optional argument containing a specific commit hash, branch name or tag

## Cloning from GitHub

The `skyllh` (and an optional private [i3skyllh](#i3skyllh)) package can be installed by cloning the GitHub repository and adding it to the Python path:

```python
import sys

sys.path.insert(0, '/path/to/skyllh')
sys.path.insert(0, '/path/to/i3skyllh')  # optional
```

# Development

We are using [`black`](https://black.readthedocs.io/en/stable/) for code linting and have configured pre-commit hooks.

To pass automatic tests on GitHub we recommend to install `pre-commit` package:
```bash
pip install pre-commit
```
and run:
```bash
pre-commit install
```
inside the cloned repository once. The pre-commit hook will then automatically run before any commit and will show `Passed`/`Failed` tests. In case of `Failed` test you can review automatic changes using `git diff`, add updated files using `git add -u`, and commit again.

It will also show existing `flake8` errors, but treat them as warnings. Feel free to fix them!

# Publications

Several publications about the SkyLLH software are available:

- IceCube Collaboration, T. Kontrimas, M. Wolf, et al. PoS ICRC2021 (2022) 1073
  [DOI](http://doi.org/10.22323/1.395.1073)
- IceCube Collaboration, M. Wolf, et al. PoS ICRC2019 (2020) 1035
  [DOI](https://doi.org/10.22323/1.358.1035)

# i3skyllh

The [`i3skyllh`](https://github.com/icecube/i3skyllh) package provides complementary pre-defined common analyses and datasets for the [IceCube Neutrino Observatory](https://icecube.wisc.edu) detector in a private [repository](https://github.com/icecube/i3skyllh).