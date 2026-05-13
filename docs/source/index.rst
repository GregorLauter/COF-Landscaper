COF-Landscaper
===============

GitHub: https://github.com/GregorLauter/COF-Landscaper

COF-Landscaper is a Python package for building and analyzing two-dimensional
covalent organic frameworks (COFs).

It provides tools to construct single-layer COF structures from node and linker
building blocks, generate interlayer-distance and interlayer-slipping structure
matrices, run machine-learning interatomic potential workflows, analyze stacking
minima, and compare simulated and experimental PXRD patterns.

Installation
------------

COF-Landscaper requires Python 3.12.

Install COF-Landscaper from PyPI:

.. code-block:: bash

   pip install cof-landscaper

Install PORMAKE, which is required for COF construction:

.. code-block:: bash

   pip install "pormake @ git+https://github.com/Sangwon91/PORMAKE.git"

To copy the example workflows into the current directory, run:

.. code-block:: bash

   cof-landscaper-copy-examples

Developer Setup
---------------

Install `just <https://github.com/casey/just>`_.

Install `uv <https://docs.astral.sh/uv/>`_.

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

Overview
--------

COF-Landscaper is designed around a complete computational workflow for
screening 2D COF stacking configurations.

COF Construction
~~~~~~~~~~~~~~~~

COF-Landscaper builds single-layer COF structures from user-provided node and
linker fragments in ``.xyz`` format. The currently supported topology keys are
``hcb``, ``sql``, ``hcb_ab``, and ``kgm``.

The ``hcb``, ``sql``, and ``kgm`` topologies require one node and one linker.
The ``hcb_ab`` topology requires two nodes and no linker.

ILD/ILS Matrix Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

The package generates structure matrices by scanning interlayer distance (ILD)
and interlayer slipping (ILS). Both serrated and inclined stacking variants can
be generated. The default slip endpoint corresponds to the AB/default stacking
shift and can be printed from the workflow settings.

MLIP Workflow
~~~~~~~~~~~~~

The MLIP workflow can be run locally using MACE-based single-point and
optimization steps.

PXRD Analysis
~~~~~~~~~~~~~

COF-Landscaper can simulate PXRD patterns, extract peaks, and compare simulated
PXRD data with experimental PXRD data.

Example Workflows
-----------------

Example files can be copied after installation with:

.. code-block:: bash

   cof-landscaper-copy-examples

The copied examples include an executable Python workflow with a separate
``cof-landscaper.params.json`` configuration file, as well as notebook workflows
with different levels of explanation.

The notebook examples include:

- ``cof-landscaper_configurable.ipynb``: full notebook with Markdown
  explanations for all configurable options.
- ``cof-landscaper.ipynb``: default workflow notebook with explanations for the
  default settings.
- ``cof-landscaper_minimal.ipynb``: minimal code-only workflow for running the
  notebook without extended explanations.

What Next?
----------

New users should start by installing COF-Landscaper, copying the example files,
and running one of the notebooks or the Python workflow in ``examples/python/``.

For API-level details, see the module documentation.

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