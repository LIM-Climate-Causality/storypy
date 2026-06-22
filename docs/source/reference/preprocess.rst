.. _preprocess:

.. |br| raw:: html

   <br />

.. |brr| raw:: html

   <br /> <br />


storypy.preprocess module
=========================

This module provides two preprocessing pathways for ingesting CMIP data into the storypy pipeline:

- **Option A** - ESMValTool-based preprocessing via :class:`ESMValProcessor`
- **Option B** - Local NetCDF-based preprocessing via :class:`ModelDataPreprocessor`

Both classes expose the same interface (``process_var`` and ``process_driver``), so downstream pipeline steps are identical regardless of which option is used.

.. note::

   Option A requires a working ESMValTool installation and a ``config-user.yml`` pointing to your local CMIP6 data pool. Option B requires that CMIP-style NetCDF files are already available locally and follow the expected directory and naming conventions described in the :ref:`overview <overview>`.

----

ESMValTool preprocessing (Option A)
------------------------------------

.. currentmodule:: storypy.preprocess._esmval_processor

.. autofunction:: parse_config

.. autoclass:: ESMValProcessor
   :members: __init__, process_var, process_driver
   :show-inheritance:

----

Local NetCDF preprocessing (Option B)
--------------------------------------

.. currentmodule:: storypy.preprocess._netcdf_processor

.. autoclass:: ModelDataPreprocessor
   :members:
   :show-inheritance: