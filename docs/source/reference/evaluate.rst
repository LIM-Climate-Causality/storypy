.. _evaluate:

storypy.evaluate.plot
=====================

Storyline construction and plotting.

.. currentmodule:: storypy.evaluate.plot

----

Storylines
----------

With two standardized drivers, a storyline is the regression response at a chosen
combination of driver values on the ellipse that contains a given fraction
(``confidence_level``) of the bivariate distribution of the drivers:

.. math::

   S_x = a_x \pm c\, b_{1,x} \pm c\, b_{2,x}

For independent drivers :math:`c = \sqrt{\chi^2_2(p)/2}` (1.27 for 80 %), as in Zappa &
Shepherd (2017). With correlated drivers (``use_correlation=True``) the coefficient differs
for storylines where the drivers move together or in opposite directions (Mindlin et al.
2020; Monerie et al. 2023); see :func:`bivariate_dist`.

.. code-block:: python

   from storypy.evaluate.plot import storyline_evaluation

   storylines, labels = storyline_evaluation(
       {"work_dir": "./output"}, target="pr", drivers=["ta", "pw"], gw_level=3.0
   )
   # labels: ['high pw low ta', 'high pw high ta', 'low pw low ta', 'low pw high ta', 'MEM']

.. autofunction:: storyline_evaluation

.. autofunction:: bivariate_dist

.. autofunction:: regression_coefficient

----

Change maps with stippling
--------------------------

:func:`plot_function` shades a projected change and adds both significance criteria:

- **filled dots**: β criterion, at least 90 % of models agree on the sign of change;
- **open circles**: γ criterion, forced signal larger than internal variability;
- **dot inside a circle**: both.

.. code-block:: python

   import xarray as xr
   from storypy.evaluate.plot import plot_function

   target_mean = xr.open_dataset("target_pr.nc")["pr"].mean("model")
   pos = xr.open_dataset("stippling/number_of_models_positive_trend_CMIP6.nc")
   neg = xr.open_dataset("stippling/number_of_models_negative_trend_CMIP6.nc")
   gamma = xr.open_dataset("stippling/gamma_forced_CMIP6.nc")["gamma"]

   # 0 where gamma > 1 (significant), 1 elsewhere
   gamma_as_pval = xr.where(gamma > 1.0, 0.0, 1.0)

   fig = plot_function(
       target_change=target_mean,
       p_values=gamma_as_pval,
       positives_model=pos,
       negatives_model=neg,
       region_extents=[(30, 45, -10, 40)],   # (lat_min, lat_max, lon_min, lon_max)
       sig_level=0.5,
       projection="robinson",               # or "platecarree" with map_extent=[...]
   )

.. autofunction:: plot_function

----

Storyline and coefficient maps
------------------------------

.. autofunction:: plot_storyline_map

.. autofunction:: create_multi_panel_figure

.. autofunction:: plot_map

.. autofunction:: make_symmetric_colorbar

----

Time series
-----------

.. autofunction:: plot_anomaly_series

``plot_precipitation_change`` is deprecated; use :func:`plot_anomaly_series`.

----

Driver scatter plots
--------------------

.. autofunction:: plot_ellipse

.. autofunction:: confidence_ellipse

.. autofunction:: create_arc
