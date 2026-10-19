.. _configuration:

Configuration
=============

storypy is configured with plain Python dicts (a YAML file read with
``storypy.confi.load_config`` works too). This page lists the keys each entry point reads.

Main configuration (``user_config``)
------------------------------------

Used by :class:`~storypy.preprocess.ModelDataPreprocessor` (Option B) and
:class:`~storypy.preprocess.ESMValProcessor` (Option A).

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Key
     - Required
     - Meaning
   * - ``work_dir``
     - yes
     - Output directory (must exist); ``target_<var>.nc`` is written here.
   * - ``plot_dir``
     - yes
     - Directory for the time-series figures (must exist).
   * - ``var_name``
     - yes
     - Target variables, e.g. ``["pr"]``.
   * - ``season``
     - yes
     - Months, e.g. ``[12, 1, 2]`` for DJF or ``[11, 12, 1, 2, 3]`` for NDJFM. A season that
       wraps across New Year counts December in the following year.
   * - ``period1`` / ``period2``
     - yes
     - Reference and future periods, e.g. ``["1960", "1990"]`` and ``["2070", "2099"]``.
   * - ``region_extents``
     - yes
     - Regions of interest ``(lat_min, lat_max, lon_min, lon_max)``. The processing box is
       their union plus 5 degrees on each side.
   * - ``region_method``
     - yes
     - ``"box"`` or ``"mask"`` (IPCC AR6 land region ``region_id``).
   * - ``region_id``
     - yes (may be ``None``)
     - AR6 region number for ``region_method="mask"``.
   * - ``data_dir``, ``exp_name``, ``freq``, ``grid``
     - Option B
     - Location and naming of the local CMIP files, e.g. ``"ssp585"``, ``"mon"``, ``"g025"``.
   * - ``variant_selection``
     - Option A, no
     - Combine ensemble members by ``"mean"`` (default), ``"first"`` or ``"last"``.
   * - ``region_labels``
     - no
     - Panel titles of the time-series plot.

Example:

.. code-block:: python

   user_config = dict(
       work_dir="/path/to/work", plot_dir="/path/to/plots",
       data_dir="/path/to/cmip6", exp_name="ssp585", freq="mon", grid="g025",
       var_name=["pr"], season=[12, 1, 2],
       period1=["1960", "1990"], period2=["2070", "2099"],
       region_method="box", region_id=None,
       region_extents=[(30, 45, -10, 40), (45, 55, 5, 20)],
       region_labels=["Mediterranean", "Central Europe"],
   )

Driver configuration (``driver_config``)
----------------------------------------

Optional second argument of the processors; missing keys fall back to ``user_config``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``var_name``
     - Variables to read, e.g. ``["tas", "psl"]``.
   * - ``short_name``
     - Output names, same length as ``var_name``. The short name ``"gw"`` is treated as
       global warming: averaged globally and used to scale the other drivers.
   * - ``box``
     - Driver region ``{"lat_min", "lat_max", "lon_min", "lon_max"}``.
   * - ``period1``, ``period2``, ``season``, ``region_extents``, ``work_dir``
     - Overrides for the drivers.

Example:

.. code-block:: python

   driver_config = dict(
       var_name=["tas", "psl", "ua"],
       short_name=["gw", "ubi", "spv"],    # 'gw' = global warming, used for scaling
       box={"lat_min": 50, "lat_max": 70, "lon_min": 40, "lon_max": 70},
       work_dir="/path/to/work/driver_outputs",
   )

Driver indices
--------------

:func:`~storypy.compute.compute_drivers` takes ``work_dir``, ``var_name``, ``short_name`` and
optionally ``gradient_drivers``: a list of ``(name, a, b)`` that adds the driver ``a - b``.

:func:`~storypy.compute.driver_indices` (ESMValTool) reads:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``input_data``, ``work_dir`` (required)
     - ESMValTool metadata and output directory.
   * - ``exclude_vars``
     - Variable groups that are targets, not drivers. Replaces the default
       :data:`~storypy.compute._compute_driver.EXCLUDE_VARS`.
   * - ``extra_exclude_vars``
     - Added to the excluded groups.
   * - ``variant_selection``
     - ``"mean"`` (default), ``"first"`` or ``"last"``.
   * - ``season_months``, ``ref_period``, ``future_period``
     - For raw monthly input: seasonal mean, anomaly against ``ref_period`` (default
       ``("1960", "1990")``) and mean over ``future_period`` (default ``("2070", "2099")``).
       Leave ``season_months=None`` when ESMValTool already averaged.
   * - ``gradient_drivers``
     - As for :func:`~storypy.compute.compute_drivers`.

.. code-block:: python

   # treat 'tos' as a target too
   driver_indices({**esmval_config, "extra_exclude_vars": ["tos"]})

Regression
----------

:func:`~storypy.compute.compute_regression`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``work_dir`` (required)
     - Reads ``<work_dir>/target_<var>.nc``, writes ``<work_dir>/regression_output/<var>/``.
   * - ``target_variable`` (required)
     - A list with exactly one variable, e.g. ``["pr"]``.
   * - ``driver_csv``
     - Explicit path of the standardized driver CSV. Default: searched under ``work_dir``.
   * - ``drivers``
     - Subset of driver columns, e.g. ``["ta", "pw"]``.

Fixed conventions
-----------------

These are set in the code and not configurable:

- Global warming is computed from ``tas`` (Option B) or read from the ``gw`` variable group
  (Option A), and the global-warming driver's short name is ``gw``.
- ``pr`` is converted from kg m-2 s-1 to mm/day (× 86400).
- The regression intercept is named ``MEM``.
- Time-series plots use the years 1950–2100, a 1960–1990 baseline, a 2070–2100
  end-of-century box and a 30-year rolling mean.
