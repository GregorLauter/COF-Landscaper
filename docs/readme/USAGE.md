---
title: "COF-Landscaper Usage Guide"
subtitle: "Workflow, parameters, examples, and outputs"
author: "Gregor Lauter"
toc: true
numbersections: true
geometry: margin=2cm
fontsize: 11pt
colorlinks: true
---

# COF-Landscaper Usage Guide

This guide explains how to run the COF-Landscaper workflow, including the notebook workflow and the hybrid script + JSON workflow. Installation steps are in [README.md](../../README.md). This document focuses on how to use the software.

> **Note:** This manual describes the current recommended workflow. The example notebook includes additional non-default settings and remains the most detailed reference for advanced usage.

## What COF-Landscaper Does

COF-Landscaper is a Python workflow for building and screening stacked 2D COF structures. It:

- builds 2D COFs from node and linker fragments
- samples stacking configurations using ILD x ILS grids
- evaluates energies with MACE and optionally DFT
- plots potential energy landscapes (PES)
- selects candidate structures for refinement
- optimizes structures (MACE or DFT)
- analyzes relaxed ILD and ILS
- simulates PXRD patterns for comparison with experiment

## Quick Start

Choose one of the two workflows below.

### Option A: Interactive notebook workflow

Use this for manual inspection, explanations, plotting, and non-default workflows.

1. Open [examples/notebook/cof-landscaper.ipynb](../../examples/notebook/cof-landscaper.ipynb).
2. Set `COF_NAME`, `TOPOLOGY`, `BOND_TYPE`, `MODE`, and `MACE_HEAD` as needed.
3. Run the notebook cells for construction, preoptimization, matrix generation, PES screening, candidate selection, optimization, analysis, and PXRD.

### Option B: Hybrid script + JSON workflow

Use this for batch execution or HPC-style runs.

1. Edit the parameter file [examples/hybrid/cof-landscaper.params.json](../../examples/hybrid/cof-landscaper.params.json), or pass a custom path via `--params`.
2. Make sure your working directory contains `0_node/` and `0_linker/` with exactly one `.xyz` file each.
3. Run the workflow script:

```bash
python examples/hybrid/cof-landscaper.py
```

To use a custom JSON path:

```bash
python examples/hybrid/cof-landscaper.py --params /path/to/params.json
```

If you want to run from the example folder where the input fragments already exist:

```bash
cd examples/hybrid
python cof-landscaper.py
```

The notebook can be used afterward to load outputs and generate PXRD comparison plots without running additional compute stages.

## Required Inputs

### Node and linker `.xyz` fragments

- The workflow expects node and linker building units of the final COF layer, not necessarily the experimental precursors.
- No explicit reaction or fragmentation step is modeled.
- Fragments should be roughly relaxed before use (for example with UFF in a molecular editor).
- Default input locations are `0_node/` and `0_linker/` in the current working directory.

> **Warning:** Provide roughly relaxed fragments to avoid severe steric clashes during stacking.

> **Warning:** Highly distorted or strongly non-planar fragments can cause clashes when the stacking matrix is generated.

### Dummy atoms and connection sites

Dummy atoms define connection sites in the fragments:

- `He` = single bond connection
- `Se` = double bond connection

> **Warning:** The dummy atom type must match `BOND_TYPE`. If these are inconsistent, COF construction may fail.

### Topology selection

Supported values for `TOPOLOGY`:

- `hcb`
- `sql`

### Stacking mode selection

Supported values for `MODE`:

- `incl` (inclined stacking)
- `serr` (serrated stacking)
- `both` (generate both)

### Optional experimental PXRD `.xy` file

Use experimental `.xy` data for `PXRD.plot_sim_vs_exp` comparisons. If `exp_xy_file` is not provided, the function searches for exactly one `.xy` file in an `experimental_pxrd` folder in the current working directory.

## Workflow Overview

1. Build a single-layer COF
2. Preoptimize the layer with constrained z motion
3. Generate the ILD x ILS stacking matrix
4. Run MACE single-point energies
5. Plot the PES
6. Select candidate structures
7. Optimize candidates with MACE or DFT
8. Analyze relaxed stacking metrics
9. Simulate PXRD and compare to experiment

![COF-Landscaper workflow overview](workflow.png){ width=85% }

## Step-by-Step Workflow Reference

### COF construction (`BuildCOF2D`)

Builds one unoptimized single-layer COF from a node and a linker. When `bond_type` is provided (the normal workflow), it reads one `.xyz` file from each of `0_node/` and `0_linker/`.

Example:

```python
builder = cl.BuildCOF2D()
builder.build(topo=TOPOLOGY, bond_type=BOND_TYPE, cof_name=COF_NAME)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `topo` | required | Topology key (`hcb` or `sql`) |
| `bond_type` | required | Bond type (`single` or `double`) |
| `cof_name` | required | COF identifier for output naming |
| `input_node` | `None` | Uses `0_node/*.xyz` when `bond_type` is set |
| `input_linker` | `None` | Uses `0_linker/*.xyz` when `bond_type` is set |
| `output_folder` | `None` | Uses `{COF_NAME}/1_{COF_NAME}_single_layer` |

Expected output:

```text
{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_unopt.cif
```

### Single-layer preoptimization (`MaceOpt.run_preopt`)

Preoptimizes the single-layer COF while constraining out-of-plane motion. This keeps the layer approximately planar so the stacking matrix does not contain severe clashes.

Example:

```python
preopt = cl.MaceOpt()
preopt.run_preopt(cof_name=COF_NAME)
```

Constructor defaults (`MaceOpt`):

| Parameter | Default | Meaning |
|---|---|---|
| `fmax` | `0.01` | Force convergence threshold in eV/A |
| `dtype` | `float64` | Numerical precision |
| `head` | `spice_wB97M` | MACE head preset |
| `model` | `None` | MACE model ID (defaults to `mh-1`) |
| `device` | `cpu` | Compute device (`cpu` or `cuda`) |
| `fix_z` | `False` | Constrain z during general optimizations |
| `max_steps` | `2000` | Maximum optimizer steps |
| `verbose` | `True` | Write MACE init logs to `mace_calculator.log` |

Method defaults (`run_preopt`):

| Parameter | Default | Meaning |
|---|---|---|
| `input_path` | `None` | `{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_unopt.cif` |
| `output_path` | `None` | `{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_preopt.cif` |
| `fix_z` | `True` | Constrain z for preoptimization |

> **Warning:** Use `fix_z=True` only for preoptimization. For bulk optimization, set `fix_z=False`.

Expected output:

```text
{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_preopt.cif
```

### Stacking matrix generation (`CreateMatrix.run`)

Generates an ILD x ILS grid by scaling the z lattice parameter (ILD) and applying in-plane slips (ILS). AA and AB are reference limits within the scan. Inclined and serrated modes are supported.

Example:

```python
matrix = cl.CreateMatrix()
matrix.run(cof_name=COF_NAME, topo=TOPOLOGY, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `ild_start` | `3.0` | Minimum ILD (Angstrom) |
| `ild_end` | `4.0` | Maximum ILD (Angstrom) |
| `ild_step` | `0.1` | ILD step size (Angstrom) |
| `ils_length_start` | `0.0` | Minimum ILS (AA reference) |
| `ils_length_end` | `None` | Auto-compute AB shift length |
| `ils_length_step` | `1.0` | ILS step size (Angstrom) |
| `ils_angle` | `None` | Auto-compute AB shift angle |
| `print_shift` | `False` | Print default AB shift values |
| `input_cif` | `None` | `{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_preopt.cif` |
| `output_base_folder` | `None` | `{COF_NAME}/2_{COF_NAME}_matrix` |

Expected output:

```text
{COF_NAME}/2_{COF_NAME}_matrix/{serr|incl}/
```

### MACE single-point energies (`MaceSP.run_mode`)

Computes single-point energies for all stacked structures in the ILD x ILS grid. No relaxation occurs here, so the PES is a reduced-dimensional screening model.

Example:

```python
sp = cl.MaceSP()
sp.run_mode(cof_name=COF_NAME, mode=MODE)
```

Constructor defaults (`MaceSP`):

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `cpu` | Compute device (`cpu` or `cuda`) |
| `dtype` | `float64` | Numerical precision |
| `head` | `spice_wB97M` | MACE head preset |
| `model` | `None` | MACE model ID (defaults to `mh-1`) |
| `verbose` | `True` | Write MACE init logs to `mace_calculator.log` |

Method defaults (`run_mode`):

| Parameter | Default | Meaning |
|---|---|---|
| `input_folder` | `None` | Uses `{COF_NAME}/2_{COF_NAME}_matrix/{serr|incl}` |
| `output_csv_dir` | `None` | Uses `{COF_NAME}/3_{COF_NAME}_landscape` |

Expected output:

```text
{COF_NAME}/3_{COF_NAME}_landscape/{COF_NAME}_sp_energies_{serr|incl}.csv
```

### Optional DFT single-point input generation (`CrystalSP.generate_input`)

Generates CRYSTAL23 `.d12` input files for the stacked structures. These jobs must be run externally (HPC).

Example:

```python
sp = cl.CrystalSP()
sp.generate_input(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `basisset` | `SOLDEF2MSVP` | CRYSTAL basis set |
| `functional` | `HSESOL3C` | Exchange-correlation functional |
| `shrink` | `2 2 8` | Monkhorst-Pack grid |
| `post_block` | `None` | Auto-generate BASISSET/DFT/SHRINK block |
| `input_base_folder` | `None` | `{COF_NAME}/2_{COF_NAME}_matrix` |
| `output_base_folder` | `None` | `{COF_NAME}/2_{COF_NAME}_matrix` |

> **Warning:** CRYSTAL23 jobs are not run by the workflow. Run them externally and place `.out` files next to their `.d12` inputs before parsing.

Expected output:

```text
{COF_NAME}/2_{COF_NAME}_matrix/dft_{serr|incl}/
```

### DFT single-point energy extraction (`CrystalSP.read_output`)

Parses CRYSTAL23 `.out` files and writes `_dft` energy CSVs for PES analysis.

Example:

```python
sp = cl.CrystalSP()
sp.read_output(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `input_base_folder` | `None` | `{COF_NAME}/2_{COF_NAME}_matrix` |
| `output_base_folder` | `None` | `{COF_NAME}/3_{COF_NAME}_landscape` |

> **Warning:** `.out` files must be placed in `dft_{serr|incl}` next to their `.d12` inputs before parsing.

Expected output:

```text
{COF_NAME}/3_{COF_NAME}_landscape/{COF_NAME}_sp_energies_{serr|incl}_dft.csv
```

### PES plotting (`Landscape.run_mode`)

Generates heatmap and/or contour plots from the single-point CSVs and marks global or local minima.

Example:

```python
landscape = cl.Landscape()
landscape.run_mode(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `colorscheme` | `viridis` | Matplotlib colormap |
| `plot_mode` | `both` | `heatmap`, `isolines`, or `both` |
| `rel_energy_max` | `None` | Clip relative energies above this value |
| `show_minima_markers` | `True` | Mark minima on the plot |
| `minima_mode` | `global` | Mark global or local minima |
| `show_header` | `True` | Draw title/header text |
| `show_title_block` | `False` | Add mode and level-of-theory lines |
| `show` | `False` | Interactive display |
| `dft` | `False` | Read `_dft` CSVs and label as DFT |
| `input_folder` | `None` | `{COF_NAME}/3_{COF_NAME}_landscape` |
| `output_folder` | `None` | `{COF_NAME}/3_{COF_NAME}_landscape` |

> **Warning:** The PES is a reduced-dimensional screening model. Always inspect structures and energies before committing to DFT refinement.

Expected output:

```text
{COF_NAME}/3_{COF_NAME}_landscape/pes_{COF_NAME}_{serr|incl}_{heatmap|isolines}.png
```

### Candidate selection (`SelectCofs.run_mode`)

Copies selected structures (global/local minima and optional manual picks) into optimization-ready folders. Use `selections_serr` and `selections_incl` to add explicit ILD/ILS points.

Example:

```python
selector = cl.SelectCofs()
selector.run_mode(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `selections_serr` | `None` | Extra ILD/ILS points for serrated mode |
| `selections_incl` | `None` | Extra ILD/ILS points for inclined mode |
| `include_autoselect` | `True` | Include auto-selected minima |
| `autoselect_minima` | `global` | Select `global` or `local` minima |
| `input_base` | `None` | `{COF_NAME}/2_{COF_NAME}_matrix` |
| `output_base` | `None` | `{COF_NAME}/3_{COF_NAME}_landscape/selection` |
| `input_folder` | `None` | Optional single-mode override |
| `output_folder` | `None` | Optional single-mode override |

Expected output:

```text
{COF_NAME}/3_{COF_NAME}_landscape/selection/{serr|incl}/
```

### Geometry optimization (`MaceOpt.run_mode`)

Optimizes candidate structures with full cell and atomic relaxation. Uses ASE `FrechetCellFilter` with `LBFGS`. Writes optimized CIFs and optional energy CSV.

Example:

```python
opt = cl.MaceOpt()
opt.run_mode(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `fmax` | `0.01` | Force convergence threshold in eV/A |
| `max_steps` | `2000` | Maximum optimizer steps |
| `dtype` | `float64` | Numerical precision |
| `head` | `spice_wB97M` | MACE head preset |
| `device` | `cpu` | Compute device (`cpu` or `cuda`) |
| `fix_z` | `False` | Do not constrain z during bulk optimization |
| `input_base` | `None` | `{COF_NAME}/3_{COF_NAME}_landscape/selection` |
| `output_base` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |
| `save_opt_energies_csv` | `True` | Write per-layer energy CSV |

Expected output:

```text
{COF_NAME}/4_{COF_NAME}_optimization/{serr|incl}/
{COF_NAME}/4_{COF_NAME}_optimization/{COF_NAME}_opt_energies_per_layer.csv
```

### Optional DFT optimization input generation (`CrystalOpt.generate_input`)

Creates CRYSTAL23 `.d12` inputs for geometry optimizations of selected structures.

Example:

```python
opt = cl.CrystalOpt()
opt.generate_input(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `basisset` | `SOLDEF2MSVP` | CRYSTAL basis set |
| `functional` | `HSESOL3C` | Exchange-correlation functional |
| `shrink` | `2 2 8` | Monkhorst-Pack grid |
| `maxtradius` | `0.5` | OPTGEOM trust radius |
| `post_block` | `None` | Auto-generate OPTGEOM/DFT blocks |
| `input_base_folder` | `None` | `{COF_NAME}/3_{COF_NAME}_landscape/selection` |
| `output_base_folder` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |

> **Warning:** CRYSTAL23 geometry optimizations are run externally. Place `.out` files next to their `.d12` inputs before extraction.

Expected output:

```text
{COF_NAME}/4_{COF_NAME}_optimization/dft_{serr|incl}/
```

### Extract DFT optimized structures (`CrystalOpt.extract_cif`, `CrystalOpt.read_output`)

Parses CRYSTAL23 `.out` files to CIFs and writes DFT energy CSVs.

Example:

```python
opt = cl.CrystalOpt()
opt.extract_cif(cof_name=COF_NAME, mode=MODE)
opt.read_output(cof_name=COF_NAME, mode=MODE)
```

Defaults (`extract_cif`):

| Parameter | Default | Meaning |
|---|---|---|
| `input_base_folder` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |
| `output_base_folder` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |

Defaults (`read_output`):

| Parameter | Default | Meaning |
|---|---|---|
| `input_base_folder` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |
| `output_base_folder` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |

> **Warning:** CRYSTAL `.out` files must be placed next to their `.d12` inputs before parsing.

Expected output:

```text
{COF_NAME}/4_{COF_NAME}_optimization/dft_{serr|incl}/
{COF_NAME}/4_{COF_NAME}_optimization/{COF_NAME}_opt_energies_per_layer_dft.csv
```

### Analyze stacking (`AnalyzeStacking.analyze`)

Computes relaxed ILD and ILS and writes summary CSVs. For serrated stacking, ILS uses a registry-based in-plane shift plus half of the tilt component. For inclined stacking, ILS is derived from the projection of the c vector onto the ab plane. ILD is halved for serrated bilayers.

Example:

```python
analyzer = cl.AnalyzeStacking()
analyzer.analyze(cof_name=COF_NAME, mode=MODE)
```

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `both` | Analyze `serr`, `incl`, or both |
| `input_base` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |
| `output_base` | `None` | `{COF_NAME}/5_{COF_NAME}_analysis` |
| `dft` | `False` | Read `dft_{mode}` folders when `True` |
| `print_values` | `True` | Print ILD/ILS values to stdout |

Expected output:

```text
{COF_NAME}/5_{COF_NAME}_analysis/final_structures.csv
```

### PXRD simulation and plotting (`PXRD.run`, `PXRD.plot_sim`, `PXRD.plot_sim_vs_exp`)

Simulates PXRD patterns from optimized CIFs and generates comparison plots.

Example (simulation):

```python
pxrd = cl.PXRD(wavelength="CuKa", two_theta_range=(1.5, 60.0))
pxrd.run(cof_name=COF_NAME, mode=MODE)
```

Defaults (`PXRD.run`):

| Parameter | Default | Meaning |
|---|---|---|
| `wavelength` | `CuKa` | X-ray wavelength preset |
| `two_theta_range` | `(1.5, 60.0)` | 2theta range (deg) |
| `mode` | `both` | `serr`, `incl`, or `both` |
| `dft` | `False` | Use DFT-optimized structures |
| `input_folder` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization/{serr|incl}` or `dft_{mode}` |
| `output_folder` | `None` | `{COF_NAME}/5_{COF_NAME}_analysis/pxrd_xy` or `pxrd_xy_dft` |

Defaults (`plot_sim`):

| Parameter | Default | Meaning |
|---|---|---|
| `xy_folder` | `None` | Uses `{COF_NAME}/5_{COF_NAME}_analysis/pxrd_xy` or `pxrd_xy_dft` |
| `output_folder` | `None` | Uses `{COF_NAME}/5_{COF_NAME}_analysis` |
| `xlim` | `(1.5, 60.0)` | 2theta plot bounds |
| `show` | `True` | Display plot |
| `save` | `True` | Save plot |

Defaults (`plot_sim_vs_exp`):

| Parameter | Default | Meaning |
|---|---|---|
| `exp_xy_file` | `None` | Searches `experimental_pxrd` for one `.xy` file |
| `simulated_xy_folder` | `None` | Uses `{COF_NAME}/5_{COF_NAME}_analysis/pxrd_xy/{mode}` or `pxrd_xy_dft/{mode}` |
| `output_folder` | `None` | Uses `{COF_NAME}/5_{COF_NAME}_analysis` |
| `xlim` | `(1.5, 60.0)` | 2theta plot bounds |
| `show` | `True` | Display plot |
| `save` | `True` | Save plot |

Expected output:

```text
{COF_NAME}/5_{COF_NAME}_analysis/pxrd_xy/{serr|incl}/
{COF_NAME}/5_{COF_NAME}_analysis/pxrd_xy_dft/{serr|incl}/
{COF_NAME}/5_{COF_NAME}_analysis/{COF_NAME}_sim_{serr|incl}.png
{COF_NAME}/5_{COF_NAME}_analysis/{COF_NAME}_{serr|incl|both}.png
```

### Optional visualization (`VisualizeCOF.visualize_cof`)

Visualizes optimized structures in a notebook using py3Dmol.

Defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `both` | `serr`, `incl`, or both |
| `input_base` | `None` | `{COF_NAME}/4_{COF_NAME}_optimization` |
| `dft` | `False` | Visualize DFT-optimized structures |
| `add_unit_cell` | `True` | Draw unit cell |
| `supercell_size_serr` | `(2, 2, 1)` | Supercell size for serrated mode |
| `supercell_size_incl` | `(2, 2, 2)` | Supercell size for inclined mode |

> **Warning:** `cuda` only works if a CUDA-enabled PyTorch/MACE installation is available on your system.

## Minimal Example

This is based on the hybrid example in [examples/hybrid/cof-landscaper.params.json](../../examples/hybrid/cof-landscaper.params.json).

Example parameters:

| Setting | Example value |
|---|---|
| `COF_NAME` | `ILCOF-1` |
| `TOPOLOGY` | `sql` |
| `BOND_TYPE` | `single` |
| `MODE` | `both` |
| `MACE_HEAD` | `spice_wB97M` |
| `DEVICE` | `cuda` |
| `MAX_STEPS` | `3000` |
| `MINIMA_MODE` | `local` |

Node and linker fragments should be placed in:

```text
examples/hybrid/0_node/
examples/hybrid/0_linker/
```

Run the workflow:

```bash
cd examples/hybrid
python cof-landscaper.py
```

Check that the first output exists:

```text
ILCOF-1/1_ILCOF-1_single_layer/ILCOF-1_unopt.cif
```

## MACE heads and dispersion settings

MACE calculations use the `MACE-MH-1` model with the selected head preset.

| `MACE_HEAD` | D3 correction | `dispersion_xc` | `dispersion_cutoff` | Notes |
|---|---|---|---:|---|
| `omat_pbe` | yes | `pbe` | `21.167088422553647` | Trained without dispersion-corrected reference data |
| `matpes_r2scan` | yes | `r2scan` | `40.0` | Trained without dispersion-corrected reference data |
| `omol` | no | N/A | N/A | Trained on dispersion-inclusive reference data |
| `spice_wB97M` | no | N/A | N/A | Trained on dispersion-inclusive reference data |

## Inputs and outputs

| Stage | Output | Location pattern |
|---|---|---|
| Single-layer construction | Unoptimized CIF | `{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_unopt.cif` |
| Preoptimization | Preoptimized CIF | `{COF_NAME}/1_{COF_NAME}_single_layer/{COF_NAME}_preopt.cif` |
| Matrix generation | Stacked CIFs | `{COF_NAME}/2_{COF_NAME}_matrix/{serr|incl}/` |
| MACE energies | CSV | `{COF_NAME}/3_{COF_NAME}_landscape/{COF_NAME}_sp_energies_{serr|incl}.csv` |
| DFT energies | CSV | `{COF_NAME}/3_{COF_NAME}_landscape/{COF_NAME}_sp_energies_{serr|incl}_dft.csv` |
| PES plots | PNGs | `{COF_NAME}/3_{COF_NAME}_landscape/pes_{COF_NAME}_{serr|incl}_{heatmap|isolines}.png` (add `_dft` suffix for DFT) |
| Candidate selection | Selected CIFs | `{COF_NAME}/3_{COF_NAME}_landscape/selection/{serr|incl}/` |
| MACE optimization | Optimized CIFs | `{COF_NAME}/4_{COF_NAME}_optimization/{serr|incl}/` |
| MACE optimization | Energy CSV | `{COF_NAME}/4_{COF_NAME}_optimization/{COF_NAME}_opt_energies_per_layer.csv` |
| DFT optimization inputs | CRYSTAL inputs | `{COF_NAME}/4_{COF_NAME}_optimization/dft_{serr|incl}/` |
| DFT optimization | Energy CSV | `{COF_NAME}/4_{COF_NAME}_optimization/{COF_NAME}_opt_energies_per_layer_dft.csv` |
| Final analysis | Summary CSV | `{COF_NAME}/5_{COF_NAME}_analysis/final_structures.csv` |
| PXRD | Simulated `.xy` | `{COF_NAME}/5_{COF_NAME}_analysis/pxrd_xy/{serr|incl}/` or `pxrd_xy_dft/{serr|incl}/` |
| PXRD plots | PNGs | `{COF_NAME}/5_{COF_NAME}_analysis/{COF_NAME}_sim_{serr|incl}.png` |

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| COF construction fails | Dummy atoms do not match `BOND_TYPE` | Ensure `He` for `single` and `Se` for `double` |
| Matrix generation creates clashes | Layer is too distorted | Preoptimize and inspect input fragments |
| No GPU acceleration | CUDA not available or `device` set to `cpu` | Install CUDA-enabled PyTorch and set `device="cuda"` |
| DFT parsing fails | `.out` files not next to `.d12` files | Move CRYSTAL outputs into the expected folder |
| PES plots missing | CSVs are not present in landscape folder | Run MACE or DFT single-point energies first |
| PXRD comparison missing | No experimental `.xy` file supplied | Provide `exp_xy_file` or add one file to `experimental_pxrd/` |

## PDF export

To generate a PDF version of this manual, run:

```bash
bash docs/readme/export_usage_pdf.sh
```

This writes `docs/readme/USAGE.pdf`. You can generate the PDF locally and commit it so users can download it directly.

## Figure placeholders

Place future images in `docs/readme/figures/` and uncomment as needed.

<!--
![Node and linker fragments with dummy atoms](figures/node_linker_dummy_atoms.png){ width=85% }
![Generated single-layer COF](figures/single_layer_unopt.png){ width=85% }
![Preoptimized single-layer COF](figures/single_layer_preopt.png){ width=85% }
![ILD x ILS matrix concept](figures/ild_ils_matrix.png){ width=85% }
![Example PES heatmap showing minima](figures/example_pes_heatmap.png){ width=85% }
![Selected minima and candidate structures](figures/selected_candidates.png){ width=85% }
![Optimized candidate structure](figures/optimized_candidate.png){ width=85% }
![Simulated vs experimental PXRD](figures/pxrd_sim_vs_exp.png){ width=85% }
-->

## Where to learn more

- Notebook workflow and explanations: [examples/notebook/cof-landscaper.ipynb](../../examples/notebook/cof-landscaper.ipynb)
- Hybrid workflow: [examples/hybrid/cof-landscaper.py](../../examples/hybrid/cof-landscaper.py) and [examples/hybrid/cof-landscaper.params.json](../../examples/hybrid/cof-landscaper.params.json)