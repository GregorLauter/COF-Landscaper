Classes
=======

The main COF-Landscaper classes are organized below according to the stages of
the workflow. For complete worked examples, see the example workflows provided
in the repository.


Structure Construction
----------------------

Classes for constructing COF structures and generating interlayer-distance
(ILD) and interlayer-slipping (ILS) structure matrices.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.BuildCOF2D
   coflandscaper.CreateMatrix
   coflandscaper.ChangeIld
   coflandscaper.IlsSerr
   coflandscaper.IlsIncl


Analysis and Visualization
--------------------------

Classes for potential-energy-landscape analysis, structure selection, stacking
analysis, and structure visualization.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.Landscape
   coflandscaper.SelectCofs
   coflandscaper.AnalyzeStacking
   coflandscaper.VisualizeCOF
   coflandscaper.Supercell


Energy Calculations and Optimization
------------------------------------

Classes for single-point energy calculations and geometry
optimization using ASE-compatible calculators.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.SinglePoint
   coflandscaper.GeometryOpt


PXRD Analysis and Refinement
----------------------------

Classes for simulated PXRD generation, comparison with experimental PXRD data,
peak analysis, and PXRD-guided structural refinement.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.PXRD


Optional CRYSTAL/DFT Workflow
-----------------------------

Classes for the optional CRYSTAL23 single-point and geometry-optimization
workflow.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.Crystal
   coflandscaper.CrystalSP
   coflandscaper.CrystalOpt