.. _evaluate:

.. |br| raw:: html

   <br />

.. |brr| raw:: html

   <br /> <br />


storypy.evaluate module
=======================

This module provides functions for visualizing projected climate change fields with stippling significance markers, region boxes, and storyline evaluation plots.

----

Plotting
--------

.. currentmodule:: storypy.evaluate._plotting

.. autofunction:: plot_function

The ``plot_function`` produces a map of projected climate change (e.g. precipitation) overlaid with significance stippling following the two-criterion approach of Zappa et al. (2021) and Mindlin et al. (2023):

- **Filled dots** - β criterion: ≥ 90% of models agree on the sign of change.
- **Open circles** - γ criterion: forced signal exceeds internal variability (γ > 1).
- **Filled dot inside open circle** - both criteria are satisfied simultaneously.

Example usage:

.. code-block:: python

   import xarray as xr
   from storypy.evaluate import plot_function

   # Load outputs from the storypy pipeline
   target_mean  = xr.open_dataset('target_pr.nc')['pr'].mean(dim='model')
   pos_ds       = xr.open_dataset('stippling/number_of_models_positive_trend_CMIP6.nc')
   neg_ds       = xr.open_dataset('stippling/number_of_models_negative_trend_CMIP6.nc')
   gamma_ds     = xr.open_dataset('stippling/gamma_forced_CMIP6.nc')

   # Convert gamma ratio to pseudo p-value (0 = significant, 1 = not significant)
   gamma        = gamma_ds['gamma']
   gamma_as_pval = xr.where(gamma > 1.0, 0.0, 1.0)

   # Global Robinson projection
   fig = plot_function(
       target_change   = target_mean,
       p_values        = gamma_as_pval,
       positives_model = pos_ds,
       negatives_model = neg_ds,
       region_extents  = [(30, 45, -10, 40)],  # (lat_min, lat_max, lon_min, lon_max)
       sig_level       = 0.5,
       sig             = 1,
       projection      = 'robinson',
   )

   # Zoomed regional PlateCarree projection
   fig = plot_function(
       target_change   = target_mean,
       p_values        = gamma_as_pval,
       positives_model = pos_ds,
       negatives_model = neg_ds,
       region_extents  = [(30, 45, -10, 40)],
       sig_level       = 0.5,
       sig             = 1,
       projection      = 'platecarree',
       map_extent      = [-30, 60, 15, 70],
   )

.. note::

   ``gamma_as_pval`` is constructed from the raw γ ratio saved in ``gamma_forced_CMIP6.nc`` by converting it to a pseudo p-value: ``gamma_as_pval = xr.where(gamma > 1.0, 0.0, 1.0)``. This ensures compatibility with the ``p_values < sig_level`` threshold used internally by ``plot_function``.

----

Storyline evaluation
--------------------

.. currentmodule:: storypy.evaluate._plotting

.. automodule:: storypy.evaluate._plotting
   :members: bivariate_dist, storyline_evaluation, regression_coefficient, make_symmetric_colorbar, create_multi_panel_figure, plot_storyline_map, plot_map, confidence_ellipse, plot_ellipse
   :show-inheritance: