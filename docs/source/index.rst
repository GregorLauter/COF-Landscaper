COF-Landscaper
===============

GitHub: https://github.com/GregorLauter/COF-Landscaper

COF-Landscaper is a Python package for building and analysing two-dimensional
covalent organic frameworks (COFs). It provides workflows for generating COF
structures from molecular building blocks, exploring stacking configurations,
and comparing simulated PXRD patterns with experimental data.

Installation
------------

COF-Landscaper requires Python 3.12.

Install COF-Landscaper from PyPI:

.. code-block:: bash

   pip install cof-landscaper

Install PORMAKE, which is required for COF construction:

.. code-block:: bash

   pip install "pormake @ git+https://github.com/Sangwon91/PORMAKE.git"


Example Workflows
-----------------

A complete example workflow is included directly in the repository under:

.. code-block:: text

   example/

The example uses one COF system and provides two complementary ways of running
COF-Landscaper.

``cof-landscaper_local.ipynb``
    Complete interactive workflow for running COF-Landscaper locally in a
    Jupyter notebook. The individual workflow steps, visualization, analysis,
    and PXRD-guided refinement can all be run directly from the notebook.

``cof-landscaper_hybrid.ipynb``
    Companion notebook for the hybrid/HPC workflow. The main computational
    workflow is executed using ``cof-landscaper.py`` on a local or HPC system,
    while the notebook is used afterwards to inspect the generated results,
    visualize the potential energy landscapes, compare simulated and
    experimental PXRD data, and perform PXRD-guided refinement.

The hybrid workflow is configured through:

``cof-landscaper.params.json``
    Contains the system and workflow parameters used by
    ``cof-landscaper.py``. The Python workflow script is intended to remain
    unchanged; calculations are configured by adjusting the JSON parameters.

Users can copy the ``example/`` directory from the repository and replace the
example inputs and parameters with their own system.


Developer Setup
---------------

Install `just <https://github.com/casey/just>`_ and
`uv <https://docs.astral.sh/uv/>`_.

Clone the repository and enter the source directory:

.. code-block:: bash

   git clone https://github.com/GregorLauter/COF-Landscaper.git
   cd COF-Landscaper

Set up the development environment:

.. code-block:: bash

   just setup

Run code checks:

.. code-block:: bash

   just check


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   Modules <modules>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`