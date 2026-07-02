"""
storypy.data
============

Utilities to load sample datasets bundled with storypy.

Quick reference
---------------

+-------------------------------------+------------------------------------------+
| Function                            | What it loads                            |
+=====================================+==========================================+
| ``read_regression(var, diagnostic)``| Pre-computed regression output           |
+-------------------------------------+------------------------------------------+
| ``load_change_field(study, var, season)`` | Pre-computed target change field         |
+-------------------------------------+------------------------------------------+
| ``list_targets()``                  | List all available target files          |
+-------------------------------------+------------------------------------------+
| ``load_driver_field(study)``          | Raw driver NetCDF for a given study      |
+-------------------------------------+------------------------------------------+
| ``read_drivers()``                  | Raw driver index CSV                     |
+-------------------------------------+------------------------------------------+
| ``read_scaled_drivers()``           | Scaled driver index CSV                  |
+-------------------------------------+------------------------------------------+
| ``read_scaled_standardized_drivers()`` | Scaled + standardized driver CSV      |
+-------------------------------------+------------------------------------------+
"""

from ._read_data import (
    read_regression,
    load_change_field,
    list_targets,
    load_driver_field,
    read_drivers,
    read_scaled_drivers,
    read_scaled_standardized_drivers,
)

__all__ = [
    "read_regression",
    "load_change_field",
    "list_targets",
    "load_driver_field",
    "read_drivers",
    "read_scaled_drivers",
    "read_scaled_standardized_drivers",
]