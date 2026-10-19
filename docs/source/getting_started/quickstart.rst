.. _quickstart:

Quickstart
==========

This example runs the regression, significance and storyline steps on the sample data
bundled with storypy, so it works right after ``pip install storypy``, without any CMIP
data. For your own data, replace the first block with :ref:`preprocessing <preprocess>`.

1. Prepare a work directory
---------------------------

storypy steps communicate through files in a ``work_dir``. Copy in the sample target (the
Zappa & Shepherd 2017 NDJFM precipitation change of 36 CMIP6 models, in mm/day per K of
global warming) and the standardized remote drivers:

.. code-block:: python

   import os
   from storypy.data import load_change_field, read_scaled_standardized_drivers

   work_dir = "storypy_quickstart"
   os.makedirs(f"{work_dir}/remote_drivers", exist_ok=True)

   target = load_change_field("zs17", "pr", "NDJFM")       # dims: model, lat, lon
   drivers = read_scaled_standardized_drivers()            # models x ['pw', 'spv', 'ta']

   target.to_netcdf(f"{work_dir}/target_pr.nc")
   drivers.to_csv(f"{work_dir}/remote_drivers/scaled_standardized_drivers.csv")

2. Regress the target on the drivers
------------------------------------

.. code-block:: python

   from storypy.compute import compute_regression

   files = compute_regression({
       "work_dir": work_dir,
       "target_variable": ["pr"],   # a list with exactly one variable
       "drivers": ["ta", "pw"],      # optional: default uses every column
   })
   # regression_output/pr/regression_coefficients.nc, ..._pvalues.nc,
   # R2.nc, ..._relative_importance.nc

3. Model agreement (β criterion)
--------------------------------

.. code-block:: python

   from storypy.compute import StipplingComputer

   pos, neg = StipplingComputer(work_dir, "pr").beta_significance(agreement_threshold=0.9)

The γ criterion also needs piControl data; see :ref:`picontrol_recipe`.

4. Storylines
-------------

.. code-block:: python

   from storypy.evaluate.plot import storyline_evaluation

   storylines, labels = storyline_evaluation({"work_dir": work_dir}, "pr", ["ta", "pw"])
   labels
   # ['high pw low ta', 'high pw high ta', 'low pw low ta', 'low pw high ta', 'MEM']

``storylines`` is a DataArray with dimensions ``storyline``, ``lat`` and ``lon``, ready
for :func:`~storypy.evaluate.plot.plot_storyline_map`. Multiply by a warming level with
``gw_level=`` to get changes in mm/day instead of mm/day/K.

Full pipeline from CMIP data
----------------------------

.. code-block:: python

   from storypy.preprocess import ModelDataPreprocessor      # or ESMValProcessor
   from storypy.compute import compute_drivers, compute_regression, StipplingComputer
   from storypy.evaluate.plot import storyline_evaluation

   proc = ModelDataPreprocessor(user_config, driver_config)
   proc.process_var()                     # -> work_dir/target_<var>.nc
   proc.process_driver()                  # -> driver work_dir/remote_driver_<name>.nc

   compute_drivers({"work_dir": driver_config["work_dir"],
                    "var_name": ["psl", "ua"], "short_name": ["ubi", "spv"]})
   compute_regression({"work_dir": user_config["work_dir"], "target_variable": ["pr"]})
   StipplingComputer(user_config["work_dir"], "pr").beta_significance()
   storylines, labels = storyline_evaluation(user_config, "pr", ["ubi", "spv"])

``user_config`` and ``driver_config`` are described in :ref:`configuration`.
