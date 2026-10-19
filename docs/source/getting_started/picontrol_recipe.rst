.. _picontrol_recipe:

piControl Recipe (γ criterion)
===============================

The γ significance criterion requires piControl simulation data to estimate internal climate variability at each grid point. This is handled by a **separate ESMValTool recipe** that runs independently of the main storypy pipeline.

.. note::

   You only need to run this recipe once per variable and season configuration. The output files are saved to ``<work_dir>/stippling/`` of the main pipeline, where you load them for :func:`~storypy.evaluate.plot.plot_function`.

----

How it works
------------

At each grid point, γ compares every model's forced response with that model's internal
variability (Zappa et al. 2021, eq. A1):

.. math::

   \gamma = \sqrt{\frac{1}{M}\sum_{m=1}^{M} \frac{\beta_m^2}{\sigma_m^2}}

where :math:`\beta_m` is the forced response of model :math:`m` (from ``target_pr.nc``) and
:math:`\sigma_m^2` is the variance of non-overlapping 30-year means of its piControl run.
A grid point is flagged (γ > 1) where the forced signal exceeds the amplitude of internal
variability. With several piControl members per model, the ratio is averaged over members
(``variant_selection='mean'``).

----

Recipe structure
----------------

Create a recipe file (e.g. ``picontrol_recipe.yml``) with the following structure:

.. code-block:: yaml

   preprocessors:
     PR:
       regrid:
         scheme:
           reference: esmf_regrid.schemes:ESMFAreaWeighted
         target_grid: 2.5x2.5

   diagnostics:
     multiple_regression_indices:
       variables:
         pr:
           exp:
           - piControl
           mip: Amon
           preprocessor: PR
           project: CMIP6
           short_name: pr
           timerange: '*'
           additional_datasets: *dAmon
       scripts:
         significance:
           script: /path/to/storypy_picontrol.py
           target_work_dir: /path/to/your/storypy/work_dir

.. note::

   ``target_work_dir`` must point to the **same** ``work_dir`` used in the main storypy pipeline, where ``target_pr.nc`` was written by ``ESMValProcessor`` or ``ModelDataPreprocessor``. This is how the diagnostic script locates the forced signal to compute γ.

   No region restriction should be applied in the preprocessor - piControl data must cover the **full global domain** for the γ computation to be valid across all longitudes.

----

Diagnostic script
-----------------

Create the diagnostic script ``storypy_picontrol.py``:

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

----

Running the recipe
------------------

Run the recipe via the ESMValTool CLI:

.. code-block:: bash

   esmvaltool run picontrol_recipe.yml

On HPC systems using SLURM, submit as a batch job:

.. code-block:: bash

   nohup esmvaltool run picontrol_recipe.yml &> picontrol_recipe.log &

----

Outputs
-------

The diagnostic saves two NetCDF files to ``{target_work_dir}/stippling/``:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File
     - Description
   * - ``gamma_forced_CMIP6.nc``
     - γ field defined above. Values > 1 indicate the forced signal exceeds
       internal variability.
   * - ``summatory_denom_CMIP6.nc``
     - Sum of :math:`1/\sigma_m^2` across models (weight used in the Mindlin et
       al. 2023 formula).

----

Using the outputs in the notebook
----------------------------------

Once the recipe has completed, load the γ output and convert it to the pseudo p-value format expected by :func:`~storypy.evaluate.plot.plot_function`:

.. code-block:: python

   import xarray as xr

   gamma_ds  = xr.open_dataset('stippling/gamma_forced_CMIP6.nc')
   gamma     = gamma_ds['gamma']

   # Convert: 0 where gamma > 1 (significant), 1 elsewhere
   gamma_as_pval = xr.where(gamma > 1.0, 0.0, 1.0)

Then pass ``gamma_as_pval`` to ``plot_function`` as the ``p_values`` argument
with ``sig_level=0.5``.

----

Parameters
----------

Arguments of :class:`~storypy.compute.StipplingComputer` and
:meth:`~storypy.compute.StipplingComputer.gamma_significance`:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``target_variable``
     - ``'pr'``
     - Variable to analyse (must match the variable saved in ``target_pr.nc``)
   * - ``picontrol_config``
     - ``None`` (required for γ)
     - ESMValTool config dict, passed by ``run_diagnostic`` in the script above
   * - ``variable_group``
     - ``'pr_picontrol'``
     - Variable group name in the piControl ESMValTool recipe (the example
       script passes ``'pr'`` to match the recipe above)
   * - ``season_months``
     - ``None``
     - Months selected before computing rolling statistics, e.g.
       ``[11, 12, 1, 2, 3]`` for NDJFM. ``None`` uses the data as delivered
   * - ``variant_selection``
     - ``'mean'``
     - How to combine multi-member piControl runs: ``'mean'`` averages over
       all variants, ``'first'`` / ``'last'`` use one
   * - ``rolling_window``
     - ``30``
     - Window length in years for the rolling mean used to estimate internal
       variability

----

Known issues
------------

**piControl data in 0-360° longitude convention**

Some CMIP6 models provide piControl output on a 0-360° longitude grid. After ESMValTool regridding the files may remain in 0-360° convention. If the forced signal (``target_pr.nc``) is in -180/180° convention, a naive ``interp_like`` call will silently produce NaN values across the entire western hemisphere (lon < 0°), resulting in γ = 0 west of 0° and a spurious 0° divide in the stippling plot.

``gamma_significance()`` automatically detects and corrects this by re-wrapping piControl longitudes to -180/180° **before** the ``interp_like`` call. No manual intervention is required.

**Short piControl runs**

Models with fewer than ``2 × rolling_window`` (60) years of seasonal data after month selection are automatically skipped with a warning. Check the ESMValTool log if fewer models than expected contribute to the γ output.