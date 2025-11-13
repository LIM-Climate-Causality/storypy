.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


storypy.compute module
======================

ESMValTool Preprocessing
------------------------

.. currentmodule:: storypy.preprocess._esmval_processor

.. autofunction:: parse_config

.. autoclass:: ESMValProcessor
   :members: __init__, process_var, process_driver


ModelData Preprocessing
-----------------------

.. currentmodule:: storypy.preprocess._netcdf_processor

.. autoclass:: ModelDataPreprocessor
   :members:
