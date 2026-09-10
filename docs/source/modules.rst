Modules
=============

The COF-Landscaper API is organized below according to the main stages of the
workflow. For a complete worked example, see the example workflows provided in
the repository.

Structure Construction
----------------------

Tools for constructing the initial single-layer COF structure from molecular
building blocks.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.BuildCOF2D


ILD/ILS Matrix Generation
-------------------------

Tools for generating stacking structures by varying interlayer distance (ILD)
and interlayer slipping (ILS).

.. autosummary::
   :toctree: _autosummary

   coflandscaper.CreateMatrix
   coflandscaper.ChangeIld
   coflandscaper.IlsSerr
   coflandscaper.IlsIncl


Energy Landscape and Structure Selection
----------------------------------------

Tools for visualizing the simplified stacking potential energy landscape and
selecting candidate structures for subsequent full geometry optimization.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.Landscape
   coflandscaper.SelectCofs


MACE Calculations and Optimization
----------------------------------

Tools for MACE-based single-point energy calculations, single-layer
preoptimization, full geometry optimization, and fixed-cell postoptimization.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.Mace
   coflandscaper.MaceSP
   coflandscaper.MaceOpt


Structure Analysis and Visualization
------------------------------------

Tools for calculating final stacking descriptors and inspecting generated COF
structures.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.AnalyzeStacking
   coflandscaper.VisualizeCOF
   coflandscaper.Supercell


PXRD Analysis and Refinement
----------------------------

Tools for simulated PXRD generation, comparison with experimental PXRD data,
peak-region analysis, PXRD-guided in-plane lattice scaling, and subsequent
structure refinement.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.PXRD


Optional CRYSTAL/DFT Workflow
-----------------------------

Alternative CRYSTAL23 interfaces for DFT single-point calculations and geometry
optimizations.

.. autosummary::
   :toctree: _autosummary

   coflandscaper.Crystal
   coflandscaper.CrystalSP
   coflandscaper.CrystalOpt