<!--
<p align="center">
  <img src="https://github.com//connectome_innolab/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  Connectome
</h1>

<p align="center">
    <a href="https://github.com//connectome_innolab/actions?query=workflow%3ATests">
        <img alt="Tests" src="https://github.com//connectome_innolab/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/connectome">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/connectome" />
    </a>
    <a href="https://pypi.org/project/connectome">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/connectome" />
    </a>
    <a href="https://github.com//connectome_innolab/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/connectome" />
    </a>
    <a href='https://connectome.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/connectome/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh//connectome_innolab/branch/main">
        <img src="https://codecov.io/gh//connectome_innolab/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com//connectome_innolab/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

<h1 align="center">
A preprocessing and graph-based analytics tool for anomaly detection in the human connectome.
</h1>

<img label='teaser_img' src='data\readme\teaser_image.jpg'>

## Abstract
Neurological diseases and psychiatric disorders are increasingly prevalent [1][2].
Despite advanced technological possibilities to measure brain connectivity and functionality, capabilities like functional Magnetic Resonance Imaging [3] are mostly used for research and sparsely for diagnosing neurological disorders in individual patients.  
**This project aims to provide a platform for medical practitioners to detect disconnectivity in individual patient connectomes and predict the probability of a neurological disorder.**
Currently, the prototype creates connectivity matrices from pre-processed fMRI images via Yeo7 atlas [4].
Connectivity matrices are labeled and augmented with metadata (age and gender).
Then, a graph neural network [(see Model architecture)](#model-architecture), trained on c. 50,000 samples from the UK Biobank dataset [5], detects anomalies using a binary classifier.
Results are evaluated with a probability of neurological disorder (anomalous connectome), brain regions summary using Niftixxx (@Simei)


> [1] https://www.paho.org/en/enlace/burden-neurological-conditions  
> [2] https://www.who.int/news-room/fact-sheets/detail/mental-disorders  
> [3] Glover, Gary H. “Overview of functional magnetic resonance imaging.” Neurosurgery clinics of North America vol. 22,2 (2011): 133-9, vii. doi:10.1016/j.nec.2010.11.001  
> [4] Yeo, B T Thomas et al. “The organization of the human cerebral cortex estimated by intrinsic functional connectivity.” Journal of neurophysiology vol. 106,3 (2011): 1125-65. doi:10.1152/jn.00338.2011



Hello! <br>
We are Sven, Oleksandr, Simei, Zhiwei and Thomas. This repo is the result of the Innovationslabor Big Data Science at LMU Munich in the winter term 22/23.

## Product demo




## Model architecture




## 💪 Getting Started

> TODO show in a very small amount of space the **MOST** useful thing your package can do.
Make it as short as possible! You have an entire set of docs for later.

### Command Line Interface

The connectome command line tool is automatically installed. It can
be used from the shell with the `--help` flag to show all subcommands:

```shell
$ connectome --help
```

> TODO show the most useful thing the CLI does! The CLI will have documentation auto-generated
by `sphinx`.

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/connectome/) with:

```bash
$ pip install connectome
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com//connectome_innolab.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com//connectome_innolab/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>


The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com//connectome_innolab.git
$ cd connectome_innolab
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com//connectome_innolab/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com//connectome_innolab.git
$ cd connectome_innolab
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/connectome/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion minor` after.
</details>
