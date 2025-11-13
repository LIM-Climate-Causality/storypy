.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


storypy.compute module
======================

Driver preparation
------------------

.. currentmodule:: storypy.compute._compute_driver

.. automodule:: storypy.compute._compute_driver
   :members: compute_drivers_from_netcdf, driver_indices, stand_numpy, stand_pandas
   :show-inheritance:

.. note::

   Additional internal helpers (e.g. ``_collect_scalar_drivers``) are used
   internally for file selection and dataset assembly and are not part of
   the public API.


Multiple linear regression
--------------------------

.. currentmodule:: storypy.compute._mlr

.. autofunction:: run_regression


Regression core utilities
-------------------------

.. currentmodule:: storypy.compute._regres

.. autoclass:: spatial_MLR
   :members: regression_data, linear_regression, linear_regression_pvalues,
             linear_regression_R2, perform_regression,
             open_regression_coef, open_lmg_coef,
             plot_regression_coef_map, plot_regression_lmg_map
   :show-inheritance:


