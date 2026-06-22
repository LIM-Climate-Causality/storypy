.. _installing:

Installing
==========


Pip install
-----------

To install storypy, run:

.. code-block:: bash

   pip install storypy

The storypy ``pip`` package has been tested successfully with the latest versions of
its dependencies (see `pyproject.toml <https://github.com/LIM-Climate-Causality/storypy/blob/main/pyproject.toml>`_).

Developer install
-----------------

To install storypy in editable mode (recommended for development or if you want to
modify the source code):

.. code-block:: bash

   git clone https://github.com/LIM-Climate-Causality/storypy.git
   cd storypy
   pip install -e .

Conda environment
-----------------

No dedicated conda package has been created yet. ``pip install storypy`` can be used
inside a conda environment.

.. note::

   Mixing ``pip`` and ``conda`` can create dependency conflicts. We recommend installing
   as many dependencies as possible with conda first, then installing storypy with
   ``pip``, `as recommended by the Anaconda team
   <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_.

ESMValTool setup (Option A only)
---------------------------------

If you intend to use the ESMValTool-based preprocessing pathway (Option A), you will
need a working ESMValTool installation and environment **in addition** to storypy.
ESMValTool is not installed automatically as a storypy dependency.

The recommended approach on HPC systems (e.g. DKRZ Levante, institute clusters) is to
load ESMValTool via the module system:

.. code-block:: bash

   module load esmvaltool

Or install it in a dedicated conda environment:

.. code-block:: bash

   conda create -n esmvaltool -c conda-forge esmvaltool
   conda activate esmvaltool
   pip install storypy

You will also need a ``config-user.yml`` pointing to your local CMIP6 data pool. See the
`ESMValTool documentation <https://docs.esmvaltool.org/en/latest/quickstart/configure.html>`_
for details.

.. note::

   Option B (local NetCDF preprocessing) does not require ESMValTool and can be used
   with any Python environment that satisfies the storypy dependencies.

Troubleshooting
---------------

Python 3.10 or above is required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

storypy requires Python 3.10 or above. Depending on your installation,
you may need to substitute ``pip`` with ``pip3``.

Build backend error on older systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter a ``BackendUnavailable`` error when installing from source, ensure
your setuptools is up to date:

.. code-block:: bash

   pip install --upgrade setuptools wheel
   pip install -e .