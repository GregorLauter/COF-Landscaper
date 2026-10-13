PXRD reflection metadata and labels
===================================

PXRD generation automatically writes both the existing two-column ``.xy`` file
and a companion ``<structure>.reflections.json``. This requires no new simulation
arguments and applies to both stacking modes, DFT outputs, postoptimization,
and the shared ``PXRD3D`` workflow. Unoptimized CIFs passed explicitly to
``produce_xy()`` receive the same metadata; there is no separate unoptimized
PXRD calculator in the current implementation.

Usage
-----

.. code-block:: python

   import coflandscaper as cl

   pxrd = cl.PXRD(wavelength="CuKa", two_theta_range=(3, 40))
   pxrd.run(cof_name="MY-COF", mode="both")
   tables = pxrd.extract_peaks(cof_name="MY-COF", mode="both")
   pxrd.plot_sim_vs_exp(
       cof_name="MY-COF", mode="both", xlim=(3, 40),
       annotate_hkl=True, max_hkl_labels=10, min_hkl_intensity=5.0,
   )
   pxrd.plot_sim(
       cof_name="MY-COF", mode="serr", annotate_hkl=True,
   )

The same options are available on ``PXRD3D.plot_sim()`` and
``PXRD3D.plot_sim_vs_exp()``. For postoptimized structures, pass
``source="postopt"`` to the 2D generation, extraction and plotting methods.
Use ``produce_xy(input_folder=..., output_folder=...)`` for a custom CIF folder.

Labels are disabled by default. The threshold is a percentage of the strongest
reflection in the **full** simulated pattern, even when the displayed interval
is narrower. The strongest eligible visible peaks are labelled, up to the
configured limit. Set ``max_hkl_labels=0`` to disable labels. Every contributing
family is displayed in classical three-index plane notation. Valid hexagonal
``(h, k, i, l)`` indices satisfy ``i = -(h + k)`` and are displayed as ``(h, k, l)``.
This is a plane-index conversion, not a direction-index transformation. Negative
indices retain their sign and are rendered with an overbar. Single-digit labels
are compact, for example ``(100)``; multi-digit labels retain commas to avoid
ambiguity. Invalid four-index relations or unexpected index sequences trigger a
warning and retain their original, explicitly signed labels. Very crowded
patterns may require reducing the label count or narrowing the angular interval.

Files and interpretation
------------------------

.. code-block:: text

   MY-COF/5_MY-COF_analysis/
       pxrd_xy/serr/
           structure.xy
           structure.reflections.json
       pxrd_peaks/serr/
           structure_all.csv
       pxrd_plots/serr/
           structure.pdf

Simulation-only plots keep their existing ``pxrd_plots/simulated/<mode>`` path.
Postoptimized outputs use ``7_MY-COF_postanalysis``. No existing paths change.

The JSON schema (version 1) records:

* Source CIF path, filename and SHA-256 fingerprint.
* Calculation settings: wavelength, angular range, scaling and symmetry tolerance.
* Indexing cell lengths, angles and lattice matrix.
* XY filename, SHA-256 fingerprint and row count.
* Each reflection's zero-based ``peak_index``, full-precision ``two_theta_deg``,
  ``relative_intensity``, ``d_spacing_angstrom``, and all ``families`` with
  ``hkl`` and ``multiplicity``.

The ``_all.csv`` tables retain the four original columns and append
``d_spacing_angstrom``, ``hkls`` and ``multiplicities``. The last two fields
contain JSON lists with matching order. Multiple families may contribute to a
single powder peak; do not interpret the combined peak intensity as a separate
intensity for each family. Metadata remains attached by XY row identity, rather
than rounded peak position, during filtering and ranking.

Indices refer to the **calculation cell**. A change of basis or supercell can
change Miller labels for the same physical reflection. Pymatgen may also return
four-index notation for hexagonal cells; all original indices remain unchanged
in JSON metadata and CSV exports. Only plot labels use the three-index format.
The metadata does not assign experimental peaks or identify a topology.

Legacy files and integrity
--------------------------

XY-only files remain supported. Their three metadata columns are unavailable
(blank in CSV, missing in the DataFrame), and requested plot labels are skipped
with a warning. There is no automatic recalculation or guessing of indices.
Regenerate PXRD from the existing CIFs to obtain metadata; structural
optimization does not need repeating.

A changed XY file, invalid metadata, or a changed source CIF at its recorded
path causes a ``ValueError`` when extracting metadata or requesting labels.
If the original CIF path is unavailable after moving or sharing a dataset,
metadata can still be read against the matching XY fingerprint; its recorded
source fingerprint and indexing basis are retained, but the source file cannot
be independently rechecked. Re-export edited XY files with fresh metadata rather
than manually changing row order or the file fingerprint.
