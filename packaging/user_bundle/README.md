# COF-Landscaper examples

## Installation

1. Install [Visual Studio Code](https://code.visualstudio.com/). Download the COF-Landscaper examples ZIP from the matching release and extract it.
2. In VS Code, choose **File → Open Folder** and open the extracted folder containing this guide and `pyproject.toml`. Accept the workspace trust prompt if you trust the download.
3. Install the recommended **Python** and **Jupyter** extensions from Microsoft. If the prompt is missing, open Extensions and search for `@recommended`.
4. Choose **Terminal → New Terminal**. On macOS or Linux, paste the following command and press Enter.

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   On Windows with WinGet, enter `winget install --id=astral-sh.uv -e`. Native Windows execution has only received a dependency-resolution check.

5. Open a fresh terminal in the same VS Code window and enter:

   ```sh
   uv sync
   ```

   Wait for installation to finish. The first run needs internet access and downloads Python 3.12 and the notebook environment into the places uv manages. The project environment is named `.venv`. If the terminal cannot find `uv`, follow the installer's PATH instructions or fully quit and reopen VS Code.

## Running the notebooks

Open [the ILCOF-1 default notebook](examples/main_workflow/local/default/default.ipynb). Click **Select Kernel → Select Another Kernel → Python Environments** and choose the Python 3.12 environment labelled **.venv** inside the extracted folder. If it is missing, finish `uv sync`, then use **Developer: Reload Window** in the Command Palette and select the kernel again. The example's saved author-specific kernel name can be ignored.

Saved outputs can be inspected immediately. A new calculation can also require a model download. The [configurable notebook](examples/main_workflow/local/configurable/configurable.ipynb), [HPC driver and analysis notebook](examples/main_workflow/hpc/), [3D workflow](examples/additional/3d/) and [optional scaling notebook](examples/additional/scaling/scaling.ipynb) are included alongside their tracked data. The HPC analysis notebook expects results from a completed HPC run. See [data sources](examples/DATA_SOURCES.md) for attribution.

For a Linux environment on Windows, open the extracted folder in a VS Code WSL window, install the recommended extensions in WSL, and follow the Linux steps there. The optional porosity backend is available on macOS and Linux.
