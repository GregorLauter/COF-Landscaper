# COF-Landscaper

COF-Landscaper is a Python package for building and analyzing 2D COF stacking-energy landscapes.

## Platform Support

- Tested on macOS and Linux.
- Microsoft Windows is currently not tested.

## Install From Source (PyPI release planned)

Create a virtual environment with Python 3.12.

```bash
python3.12 -m venv test-coflandscaper
```

Activate the environment.

```bash
source test-coflandscaper/bin/activate
```

Confirm the active Python executable.

```bash
which python
```

Confirm the Python version is 3.12.

```bash
python --version
```

Upgrade pip.

```bash
pip install --upgrade pip
```

Confirm pip is available.

```bash
pip --version
```

Clone the repository.

```bash
git clone https://github.com/GregorLauter/COF-Landscaper.git
```

Enter the project directory.

```bash
cd COF-Landscaper
```

Install the package.

```bash
pip install .
```

Install the Jupyter kernel package.

```bash
pip install ipykernel
```

Register this environment as a Jupyter kernel.

```bash
python -m ipykernel install --user --name test-coflandscaper --display-name "Python (test-coflandscaper)"
```

## Workflow Notes

- The DFT workflow requires additional external HPC infrastructure.
- The MACE workflow can be executed fully on a local machine.
- Workflow diagram (PDF): [docs/readme/workflow_vertical.pdf](docs/readme/workflow_vertical.pdf)

## Example Notebook

- Example notebook location in this repository: `examples/COF-1/0_all/cof-landscaper.ipynb`
- After installation, you can work from any project folder on your computer.
- A practical workflow is to copy the example notebook into your own project directory and keep the original examples folder as a reference.

## VS Code Recommendation

VS Code is recommended for running and editing the notebook and Python code.

To use the correct kernel in VS Code:

1. Open the notebook.
2. Click the kernel selector in the top-right.
3. Choose `Python (test-coflandscaper)`.
4. Run a test cell such as `import coflandscaper`.

## Where To Find Explanations

- A stepwise explanation of the computational workflow is provided in the Markdown cells of the example notebook.
- Methodological details, assumptions, and validation context are documented in the accompanying manuscript [insert link here].
