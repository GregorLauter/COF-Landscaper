# COF-Landscaper

COF-Landscaper is a Python package for building and analyzing 2D COFs.

Researchers interested in applying COF-Landscaper to their own systems are welcome to contact me at gjl342@student.bham.ac.uk. Depending on availability and the scope of the project, I may be able to provide support or explore a possible collaboration.

## Platform Support

- Tested on macOS and Linux.
- Microsoft Windows is currently not tested.

## Installation

COF-Landscaper requires Python 3.12.

Create and activate a virtual environment.

```bash
python3.12 -m venv test-coflandscaper
source test-coflandscaper/bin/activate
```

Upgrade pip.

```bash
pip install --upgrade pip
```

Install COF-Landscaper from PyPI.

```bash
pip install cof-landscaper
```

Install PORMAKE, which is required for COF construction.

```bash
pip install "pormake @ git+https://github.com/Sangwon91/PORMAKE.git"
```

## Running the Notebooks

Install Jupyter support if you want to run the notebooks.

```bash
pip install jupyter ipykernel
```

Register the environment as a Jupyter kernel.

```bash
python -m ipykernel install --user --name test-coflandscaper --display-name "Python (test-coflandscaper)"
```

In VS Code or Jupyter, select the kernel:

```text
Python (test-coflandscaper)
```

Run a test cell:

```python
import coflandscaper as cl
```

## Example Files

After installation, COF-Landscaper can be imported and used directly in your own Python scripts or notebooks.

If you want to start from the provided example workflows, run:

```bash
cof-landscaper-copy-examples
```

This copies the example files into the current directory under:

```text
examples/
```

The copied examples include an executable Python workflow under:

```text
examples/python/
```

This folder contains the workflow script and a separate `cof-landscaper.params.json` file where the workflow settings can be configured. It also includes a minimal notebook for plotting simulated PXRD data together with experimental PXRD data after the workflow has finished.

The copied examples also include three notebook versions under:

```text
examples/notebook/
```

The notebook versions are:

- `cof-landscaper_configurable.ipynb`: full notebook with Markdown explanations for all configurable options.
- `cof-landscaper_default.ipynb`: default workflow notebook with explanations for the default settings.
- `cof-landscaper_minimal.ipynb`: minimal code-only workflow for running the notebook without extended explanations.

You can then edit the copied Python script, JSON parameter file, notebook, and input `.xyz` files for your own system.

## Developer Setup

Install `just`.

Install `uv`.

Clone the repository and enter the source directory.

```bash
git clone https://github.com/GregorLauter/COF-Landscaper.git
cd COF-Landscaper
```

Set up the development environment.

```bash
just setup
```

Run code checks.

```bash
just check
```

## Workflow Notes

- The DFT workflow requires additional external HPC infrastructure.
- The MLIP workflow can be executed fully on a local machine.
- Workflow diagram:

![COF-Landscaper workflow](docs/readme/workflow.png)

## Required Input Files

The workflow requires building-block fragments provided as `.xyz` files.

Supported topologies are:

- `hcb`
- `sql`
- `hcb_ab`
- `kgm`

Input requirements:

- `hcb`, `sql`, and `kgm` require one node `.xyz` file and one linker `.xyz` file.
- `hcb_ab` requires two node `.xyz` files and no linker file.
- By default, node files are read from `0_node/`.
- By default, linker files are read from `0_linker/` when required by the topology.
- Explicit paths can be provided with `input_nodes=[...]` and `input_linkers=[...]`.

Input fragments should ideally be pre-optimized with a generic force field, such as UFF, to remove severe steric clashes and obtain reasonable approximate bond lengths.

The subsequent pre-optimization step handles the assembled framework. Therefore, the main requirement at this stage is that the individual fragments are chemically sensible and can be connected cleanly by the builder.

The `.xyz` files can be prepared using any suitable molecular editor or visualizer, for example Avogadro, Mercury, or DrawMol.

## Where To Find Explanations

The full documentation is available on Read the Docs:

[COF-Landscaper documentation](https://cof-landscaper.readthedocs.io/)

Additional stepwise explanations of the computational workflow are provided in the Markdown cells of the example notebooks.