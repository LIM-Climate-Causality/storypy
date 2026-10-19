.. _preprocess:

storypy.preprocess
==================

Two pathways turn CMIP output into the inputs of the storyline analysis:

- **Option A**, :class:`~storypy.preprocess.ESMValProcessor`: ESMValTool downloads,
  regrids and preprocesses the data; storypy reads the ESMValTool metadata.
- **Option B**, :class:`~storypy.preprocess.ModelDataPreprocessor`: storypy reads
  CMIP-style NetCDF files from a local directory.

Both expose ``process_var()`` (writes ``target_<var>.nc``: the change of each model,
normalised by its global warming) and ``process_driver()``, so later steps are the same.
All configuration keys are listed in :ref:`configuration`.

.. currentmodule:: storypy.preprocess

----

Option A: ESMValTool
--------------------

Only :func:`parse_config` needs ESMValTool installed. The processor itself accepts any
dict whose ``input_data`` follows ESMValTool's metadata layout.

.. autofunction:: parse_config

.. autoclass:: ESMValProcessor
   :members: process_var, process_driver

----

Option B: local NetCDF files
----------------------------

Expected layout (split on ``_``: model, experiment and variant are the 3rd, 4th and 5th
fields, as in standard CMIP file names)::

    <data_dir>/<var>/<freq>/<grid>/<var>_<table>_<model>_<experiment>_<variant>_<grid>_<period>.nc

For example ``data/pr/mon/g025/pr_Amon_ACCESS-CM2_ssp585_r1i1p1f1_gn_201501-210012.nc``.
A model is used when it has historical and scenario runs of a common variant for every
target variable, and matching ``tas`` files for the global-warming
normalisation.

.. autoclass:: ModelDataPreprocessor
   :members: process_var, process_driver

----

Diagnostics
-----------

Building blocks used by both processors.

.. autofunction:: seasonal_data_months

.. autofunction:: clim_change

.. autofunction:: apply_region_mask

.. autofunction:: adjust_longitudes

.. autofunction:: extract_metadata

.. autofunction:: test_mean_significance
