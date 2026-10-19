.. _config_api:

Utilities and defaults
======================

Defaults and helpers shared across storypy.

Variables excluded from the driver list
---------------------------------------

.. autodata:: storypy.compute._compute_driver.EXCLUDE_VARS
   :no-value:

The variable groups that :func:`storypy.compute.driver_indices` treats as targets, not
drivers: ``u850``, ``tas``, ``sst``, ``psl``, ``pr``, ``pr_djf``, ``pr_jja``,
``u850_djf``, ``u850_jja``. Change it per run with ``config['exclude_vars']`` (replace) or
``config['extra_exclude_vars']`` (add); see :ref:`configuration`.

Locating the driver table
-------------------------

.. autofunction:: storypy.utils.find_driver_csv

Reading a YAML configuration
----------------------------

``storypy.confi.load_config(path)`` reads a YAML file into a dict that can be passed to the
functions above. It needs PyYAML (``pip install pyyaml``).
