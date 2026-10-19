.. _compute:

storypy.compute
===============

Remote driver indices, spatial multiple linear regression and significance (stippling)
masks, following Zappa & Shepherd (2017), Zappa et al. (2021) and Mindlin et al. (2023).

.. currentmodule:: storypy.compute

.. autosummary::
   :nosignatures:

   compute_drivers
   driver_indices
   compute_regression
   SpatialRegression
   StipplingComputer

----

Driver indices
--------------

Two entry points produce the same three tables in ``<work_dir>/remote_drivers/``:
``drivers.csv`` (raw), ``scaled_drivers.csv`` (divided by global warming) and
``scaled_standardized_drivers.csv`` (z-scored across models, the regression input), plus
the ensemble spread in ``drivers_errors.csv`` / ``scaled_drivers_errors.csv``.

- :func:`compute_drivers` reads the NetCDF files written by
  :class:`~storypy.preprocess.ModelDataPreprocessor` (Option B).
- :func:`driver_indices` reads ESMValTool metadata (Option A). Variable groups listed in
  ``EXCLUDE_VARS`` (the targets ``u850``, ``tas``, ``sst``, ``psl``, ``pr``, ``pr_djf``,
  ``pr_jja``, ``u850_djf``, ``u850_jja``) are skipped; ``config['exclude_vars']`` replaces
  that list and ``config['extra_exclude_vars']`` adds to it. See :ref:`configuration`.

.. autofunction:: compute_drivers

.. autofunction:: driver_indices

.. autofunction:: stand_pandas

.. autofunction:: stand_numpy

----

Spatial regression
------------------

:func:`compute_regression` aligns ``target_<var>.nc`` with the standardized driver table
by model name and fits, at every gridpoint,

.. math::

   P_{xm} = a_x + \sum_k b_{k,x}\, X'_{k,m} + e_{xm}

writing four files to ``<work_dir>/regression_output/<var>/``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Content
   * - ``regression_coefficients.nc``
     - :math:`a_x` (named ``MEM``) and one :math:`b_{k,x}` per driver
   * - ``regression_coefficients_pvalues.nc``
     - two-sided t-test p-value of each coefficient
   * - ``regression_coefficients_relative_importance.nc``
     - LMG relative importance of each driver (shares of :math:`R^2`)
   * - ``R2.nc``
     - coefficient of determination

Models that are NaN at a gridpoint (e.g. 850 hPa below ground) are left out of the fit at
that gridpoint only; with fewer valid models than coefficients + 1 the result is NaN.

.. autofunction:: compute_regression

.. autoclass:: SpatialRegression
   :members: regression_data, perform_regression, linear_regression,
             linear_regression_pvalues, linear_regression_R2,
             linear_regression_relative_importance, open_regression_coef,
             open_lmg_coef, plot_regression_coef_map

.. autofunction:: storypy.compute._regres.lmg

----

.. _gamma_significance:

Significance (stippling)
------------------------

:class:`StipplingComputer` implements the two robustness criteria of Zappa et al. (2021)
and Mindlin et al. (2023):

- **β criterion**, :meth:`~StipplingComputer.beta_significance`: at least 90 % of models
  agree on the sign of the change. Works from ``target_<var>.nc`` alone.
- **γ criterion**, :meth:`~StipplingComputer.gamma_significance`: the forced signal exceeds
  internal variability estimated from piControl runs (γ > 1). Needs a separate piControl
  ESMValTool run; see :ref:`picontrol_recipe`.

Results are written to ``<work_dir>/stippling/``.

.. autoclass:: StipplingComputer
   :members: beta_significance, gamma_significance

----

Helpers
-------

.. autofunction:: stand

.. autofunction:: stand_detr

.. autofunction:: filtro

.. autofunction:: replace_nans_with_zero
