.. _data:

storypy.data module
===================

This module provides functions for loading sample datasets bundled with storypy, including pre-computed regression outputs, target climate change fields, spatial driver fields, and tabular driver indices.

All data files are stored under ``storypy/data/`` and accessed via :mod:`importlib.resources`. No file paths are required - all datasets are loaded by name using the functions below.

----

Directory layout
----------------

.. code-block:: text

    storypy/data/
    ├── drivers/                           # Spatial driver NetCDF files per study
    │   ├── zs17_drivers.nc
    │   └── monerie23_drivers.nc
    ├── regression_output/                 # Pre-computed regression results
    │   ├── pr/
    │   │   ├── regression_coefficients.nc
    │   │   ├── regression_coefficients_pvalues.nc
    │   │   ├── regression_coefficients_relative_importance.nc
    │   │   └── R2.nc
    │   └── ua/
    │       ├── regression_coefficients.nc
    │       ├── regression_coefficients_pvalues.nc
    │       ├── regression_coefficients_relative_importance.nc
    │       └── R2.nc
    ├── remote_drivers/                    # Scaled/standardized driver index CSVs
    │   ├── drivers.csv
    │   ├── scaled_drivers.csv
    │   └── scaled_standardized_drivers.csv
    └── targets/                           # Pre-computed target change fields
        ├── zs17_target_pr_NDJFM.nc
        ├── zs17_target_u850_NDJFM.nc
        ├── mindlin20_target_pr_DJF.nc
        ├── mindlin20_target_pr_JJA.nc
        └── monerie23_target_pr_JAS.nc

----

Quick reference
---------------

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Function
     - Returns
     - Description
   * - :func:`load_change_field(study, variable, season) <storypy.data.load_change_field>`
     - ``xr.Dataset``
     - Pre-computed target climate change field for a given study and season
   * - :func:`load_driver_field(study) <storypy.data.load_driver_field>`
     - ``xr.Dataset``
     - Spatial driver NetCDF for a given study
   * - :func:`list_targets() <storypy.data.list_targets>`
     - ``list``
     - List all available target files bundled with storypy
   * - :func:`list_drivers() <storypy.data.list_drivers>`
     - ``list``
     - List all available driver files bundled with storypy
   * - :func:`read_regression(variable, diagnostic) <storypy.data.read_regression>`
     - ``xr.Dataset``
     - Pre-computed regression output for a given variable and diagnostic
   * - :func:`read_drivers() <storypy.data.read_drivers>`
     - ``pd.DataFrame``
     - Raw (unscaled) remote driver indices
   * - :func:`read_scaled_drivers() <storypy.data.read_scaled_drivers>`
     - ``pd.DataFrame``
     - Scaled remote driver indices (divided by global warming)
   * - :func:`read_scaled_standardized_drivers() <storypy.data.read_scaled_standardized_drivers>`
     - ``pd.DataFrame``
     - Scaled and standardized driver indices used in regression

----

Target change fields
--------------------

.. currentmodule:: storypy.data

.. autofunction:: load_change_field

.. autofunction:: list_targets

Available studies and seasons:

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Study
     - Variable
     - Season
     - Reference
   * - ``'zs17'``
     - ``'pr'``, ``'u850'``
     - ``'NDJFM'``
     - Zappa & Shepherd (2017)
   * - ``'mindlin20'``
     - ``'pr'``
     - ``'DJF'``, ``'JJA'``
     - Mindlin et al. (2020)
   * - ``'monerie23'``
     - ``'pr'``
     - ``'JAS'``
     - Monerie et al. (2023)

----

Driver fields
-------------

.. autofunction:: load_driver_field

----

Regression outputs
------------------

.. autofunction:: read_regression

Available diagnostics:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Diagnostic
     - Description
   * - ``'regression_coefficients'``
     - OLS regression coefficients for each driver at each gridpoint
   * - ``'regression_coefficients_pvalues'``
     - p-values associated with each regression coefficient
   * - ``'regression_coefficients_relative_importance'``
     - Relative importance of each driver
   * - ``'R2'``
     - Coefficient of determination at each gridpoint

Example:

.. code-block:: python

   from storypy.data import read_regression
 
   # Load precipitation regression coefficients
   coefs = read_regression('pr')
 
   # Load u850 R² field
   r2 = read_regression('ua', diagnostic='R2')
 
   # Load p-values for precipitation
   pvals = read_regression('pr', diagnostic='regression_coefficients_pvalues')

----

Driver index CSV files
----------------------

.. autofunction:: read_drivers

.. autofunction:: read_scaled_drivers

.. autofunction:: read_scaled_standardized_drivers

.. note::

   ``read_scaled_standardized_drivers()`` returns the driver indices used directly as regressors in :func:`storypy.compute._mlr.compute_regression`. The scaling divides each driver by the ensemble mean global warming :math:`\Delta T`, and standardization divides by the cross-model standard deviation so that regression coefficients are comparable across drivers.