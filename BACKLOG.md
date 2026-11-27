# Development backlog

## Make linker orientation reproducible across numerical libraries

For COF-LZU-1, fitting a linker by its two connection points leaves rotation about their axis undetermined. The fitted linker-plane normal can take either sign during alignment to the sheet. One of the three linkers can consequently finish rotated by 180°. Its atoms moved by up to 4.70 Å between constructions, while the cell, node placements and inferred bonds agreed. The effect on relaxed energies and stacking rankings remains untested.

On the same Ubuntu 22.04 runner with the same locked Python 3.12 environment, the previous release (`v2026.11.26.0`) and the patch code (`31d74f1`) produced byte-identical CIFs for each OpenBLAS setting. Both passed the original reference test with native dispatch and failed it with the Haswell, Nehalem and Sandybridge kernels. [Controlled Linux comparison](https://github.com/GregorLauter/COF-Landscaper/actions/runs/46024420092) · [Original CI failure](https://github.com/GregorLauter/COF-Landscaper/actions/runs/46022783249)

For a future release, define a reproducible convention for the orientation of two-ended linkers and verify it across these kernels. Re-enable `tests/test_buildcof2d_utils.py::test_manual_hcb_matches_reference` with the existing archived structure and tolerances once these checks pass.
