.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


storypy.preprocess module
==========================

.. automodule:: storypy.preprocess
   :members:
   :undoc-members:
   :show-inheritance:

ESMValTool Preprocessing
------------------------

.. currentmodule:: storypy.preprocess._esmval_processor

.. autoclass:: ESMValProcessor
   :members: __init__, process_var, process_driver
   :undoc-members:
   :show-inheritance:

.. autofunction:: parse_config

Modeldata Preprocessing
-----------------------

.. currentmodule:: storypy.preprocess._netcdf_processor

.. autoclass:: NetCDFProcessor
   :members: __init__, process_var, process_driver
   :undoc-members:
   :show-inheritance:
