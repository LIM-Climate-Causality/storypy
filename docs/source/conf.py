# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# docs/source/conf.py
import os
import sys
from datetime import datetime

# --- Paths --------------------------------------------------------------------
# Assuming this file lives in docs/source/, put the repo root on sys.path
sys.path.insert(0, os.path.abspath("../.."))

# --- Project info -------------------------------------------------------------
project = "storypy"
author = "Leipzig Institute for Meteorology (LIM)"
copyright = f"{datetime.now():%Y}, {author}"
release = ""   # e.g. "1.0.0" if you want it shown
version = release

# --- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# If your modules import heavy/optional deps, mock them so RTD can import
autodoc_mock_imports = [
    "esmvaltool",
    "xarray", "netCDF4", "cartopy",
    "dask", "cfgrib", "cftime", "zarr", "eccodes",
    "numpy", "pandas", "scipy", "sklearn",
    "matplotlib", "xesmf", "yaml", "pooch",
    "statsmodels",
]

# Generate autosummary stub pages for modules/classes/functions
autosummary_generate = True

# Autodoc defaults (sane, readable API pages)
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Type hints—keep signatures clean but show hints in the description block
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"

autoclass_content = "both"

# Napoleon (NumPy/Google style) settings
napoleon_google_docstring = False   # you’re likely using NumPy style
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True


templates_path = ["_templates"]
exclude_patterns = []
todo_include_todos = False

# --- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "titles_only": False,
}

def setup(app):
    import importlib, traceback
    targets = [
        "storypy.preprocess._esmval_processor",
        "storypy.preprocess._netcdf_processor",
    ]
    for mod in targets:
        try:
            importlib.import_module(mod)
            print(f"[autodoc] import ok: {mod}")
        except Exception as e:
            print(f"[autodoc] IMPORT FAILED: {mod} -> {e}")
            traceback.print_exc()
