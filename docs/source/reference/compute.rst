.. _compute:

.. |br| raw:: html

   <br />

.. |brr| raw:: html

   <br /> <br />


storypy.compute module
======================

This module provides functions and classes for computing remote driver indices, running multiple linear regression, and performing stippling significance tests following Zappa et al. (2021) and Mindlin et al. (2023).

----

Driver preparation
------------------

.. currentmodule:: storypy.compute._compute_driver

.. automodule:: storypy.compute._compute_driver
   :members: compute_drivers_from_netcdf, driver_indices, stand_numpy, stand_pandas
   :show-inheritance:

.. note::

   Additional internal helpers (e.g. ``_collect_scalar_drivers``) are used internally for file selection and dataset assembly and are not part of the public API.

----

Multiple linear regression
--------------------------

.. currentmodule:: storypy.compute._mlr

.. autofunction:: run_regression

----

Regression core utilities
-------------------------

.. currentmodule:: storypy.compute._regres

.. autoclass:: spatial_MLR
   :members: regression_data, linear_regression, linear_regression_pvalues,
             linear_regression_R2, perform_regression,
             open_regression_coef, open_lmg_coef,
             plot_regression_coef_map, plot_regression_lmg_map
   :show-inheritance:

----

Stippling significance
----------------------

.. _gamma_significance:

.. currentmodule:: storypy.compute._stippling

.. autoclass:: StipplingComputer
   :members: beta_significance, gamma_significance
   :show-inheritance:

The ``StipplingComputer`` class implements two independent significance criteria following Zappa et al. (2021) and Mindlin et al. (2023):

- **β criterion** — computed by :meth:`beta_significance`. Flags gridpoints where ≥ 90% of CMIP6 models agree on the sign of the projected change.

- **γ criterion** — computed by :meth:`gamma_significance`. Flags gridpoints where the forced signal exceeds internal variability estimated from piControl simulations (γ > 1), using non-overlapping 30-year rolling seasonal means.

.. note::

   :meth:`gamma_significance` requires piControl data preprocessed via a separate ESMValTool recipe. The recipe passes ``target_work_dir`` to the diagnostic script so that ``StipplingComputer`` can locate ``target_pr.nc`` (the forced signal) in the same working directory as the main storypy pipeline.

   Example diagnostic script (``storypy_picontrol.py``):

   .. code-block:: python

      from esmvaltool.diag_scripts.shared import run_diagnostic
      from storypy.compute import StipplingComputer

      def main(config):
          sc = StipplingComputer(
              work_dir          = config['target_work_dir'],
              target_variable   = 'pr',
              picontrol_config  = config,
              variable_group    = 'pr',
              season_months     = [11, 12, 1, 2, 3],  # NDJFM
              variant_selection = 'mean',
          )
          sc.gamma_significance(rolling_window=30)

      if __name__ == '__main__':
          with run_diagnostic() as config:
              main(config)

   The script is run via:

   .. code-block:: bash

      esmvaltool run --config_file /path/to/config-user.yml /path/to/picontrol_recipe.yml

   Outputs are saved to ``{work_dir}/stippling/``:

   - ``gamma_forced_CMIP6.nc`` — median γ ratio across models
   - ``summatory_denom_CMIP6.nc`` — sum of 1/σ² weights

.. warning::

   piControl data must cover the full global domain. If piControl files use a 0–360° longitude convention, ``gamma_significance()`` automatically re-wraps longitudes to −180/180° before computing the signal-to-noise ratio to ensure correct spatial alignment with ``target_pr.nc``.