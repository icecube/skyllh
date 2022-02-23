# SkyLLH

[![Tests](https://github.com/icecube/skyllh/actions/workflows/pythonpackage.yml/badge.svg)](#)
[![Docs](https://github.com/icecube/skyllh/actions/workflows/documentation.yml/badge.svg)](https://icecube.github.io/skyllh/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/licenses/GPL-3.0)

[[Full documentation]](https://icecube.github.io/skyllh/).

The SkyLLH framework is an open-source Python3-based package licensed under the GPLv3 license. It provides a modular framework for implementing custom likelihood functions and executing log-likelihood ratio hypothesis tests. The idea is to provide a class structure tied to the mathematical objects of the likelihood functions, rather than to entire abstract likelihood models.

# Installation

The `skyllh` (and an optional private [i3skyllh](#i3skyllh)) package is installed by cloning the GitHub repository and adding it to the Python path:

```python
import sys

sys.path.insert(0, '/path/to/skyllh')
sys.path.insert(0, '/path/to/i3skyllh')  # optional
```

# i3skyllh

The [`i3skyllh`](https://github.com/icecube/i3skyllh) package provides complementary pre-defined common analyses and datasets for the [IceCube Neutrino Observatory](https://icecube.wisc.edu) detector in a private [repository](https://github.com/icecube/i3skyllh).