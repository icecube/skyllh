# SkyLLH

[![CI](https://github.com/icecube/skyllh/actions/workflows/ci.yml/badge.svg)](#)
[![Docs](https://github.com/icecube/skyllh/actions/workflows/documentation.yml/badge.svg)](https://icecube.github.io/skyllh/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![PyPI - Version](https://img.shields.io/pypi/v/skyllh)](https://pypi.org/project/skyllh/)
[![conda-forge](https://anaconda.org/conda-forge/skyllh/badges/version.svg)](https://anaconda.org/conda-forge/skyllh)

[[Full documentation]](https://icecube.github.io/skyllh/).

The SkyLLH framework is an open-source Python-based package licensed under the
GPLv3 license. It provides a modular framework for implementing custom
likelihood functions and executing log-likelihood ratio hypothesis tests.
The idea is to provide a class structure tied to the mathematical objects of the
likelihood functions, rather than to entire abstract likelihood models.

The math formalism used in SkyLLH is described in the math formalism
[[document]](https://github.com/icecube/skyllh/blob/master/doc/user_manual.pdf).

# Installation

Python >= 3.11 is required.

## Using pip

The latest `skyllh` release can be installed from
[PyPI](https://pypi.org/project/skyllh/) repository:
```bash
pip install skyllh
```

Optional dependency groups can be installed with extras:
```bash
pip install skyllh[extras]   # iminuit, pyarrow
pip install skyllh[dev]      # pre-commit, pytest
pip install skyllh[docs]     # sphinx and doc-build tools
```

The current development version can be installed using pip:
```bash
pip install git+https://github.com/icecube/skyllh.git
```

Optionally, a specific reference can be installed by:
```bash
pip install git+https://github.com/icecube/skyllh.git@[ref]
```
where `[ref]` is a commit hash, branch name, or tag.

## Using conda

```bash
conda install -c conda-forge skyllh
```

# Publications

Several publications about the SkyLLH software are available:

- IceCube Collaboration, C. Bellenghi, M. Karl, M. Wolf, et al. PoS ICRC2023 (2023) 1061
  [DOI](https://doi.org/10.22323/1.444.1061)
- IceCube Collaboration, T. Kontrimas, M. Wolf, et al. PoS ICRC2021 (2022) 1073
  [DOI](http://doi.org/10.22323/1.395.1073)
- IceCube Collaboration, M. Wolf, et al. PoS ICRC2019 (2020) 1035
  [DOI](https://doi.org/10.22323/1.358.1035)

# Developer Guidelines

These guidelines should help new developers of SkyLLH to join the development
process easily.

## Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com) to run [ruff](https://docs.astral.sh/ruff/) for linting and formatting on every commit.

Install `pre-commit` and set up the hooks:

```bash
pip install pre-commit
pre-commit install
```

The hooks will now run automatically on `git commit`. To run them manually against all files:

```bash
pre-commit run --all-files
```

## Branching

- When implementing a new feature / change, first an issue must be created
  describing the new feature / change. Then a branch must be created referring
  to this issue. We recommend the branch name `fix<ISSUE_NUMBER>`, where
  `<ISSUE_NUMBER>` is the number of the created issue for this feature / change.

- In cases when SkyLLH needs to be updated because of a change in the i3skyllh
  package (see below), we recommend the branch name `i3skyllh_<ISSUE_NUMBER>`,
  where `<ISSUE_NUMBER>` is the number of the issue created in the i3skyllh
  repository. That way the *analysis unit tests* workflow will be able to find
  the correct skyllh branch corresponding to the i3skyllh change automatically.

## Releases and Versioning

- Release version numbers follow the format `v<YY>.<MAJOR>.<MINOR>`, where
  `<YY>` is the current year, `<MAJOR>` and `<MINOR>` are the major and minor
  version numbers of type integer. Example: `v23.2.0`.

- Release candidates follow the same format as releases, but have the additional
  suffix `.rc<NUMBER>`,  where `<NUMBER>` is an integer starting with 1.
  Example: `v23.2.0.rc1`

- Before creating the release on github, the version number needs to be updated
  in the Sphinx documentation: `doc/sphinx/conf.py`.

# i3skyllh

The [`i3skyllh`](https://github.com/icecube/i3skyllh) package provides
complementary pre-defined common analyses and datasets for the
[IceCube Neutrino Observatory](https://icecube.wisc.edu) detector in a private
[repository](https://github.com/icecube/i3skyllh).

# Contributors

- [Martin Wolf](https://github.com/martwo) - [martin.wolf@tum.de](mailto:martin.wolf@tum.de)
- [Tomas Kontrimas](https://github.com/tomaskontrimas) - [tomas.kontrimas@tum.de](mailto:tomas.kontrimas@tum.de)
- [Chiara Bellenghi](https://github.com/chiarabellenghi) - [chiara.bellenghi@tum.de](mailto:chiara.bellenghi@tum.de)
- [Martina Karl](https://github.com/mskarl) - [martina.karl@eso.org](mailto:martina.karl@eso.org)
