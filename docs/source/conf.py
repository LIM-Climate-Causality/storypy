# Sphinx configuration for the storypy documentation.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Repo root on sys.path, so a source checkout builds without installing.
sys.path.insert(0, os.path.abspath("../.."))

# --- Project info -------------------------------------------------------------
project = "storypy"
author = "Richard Alawode, Julia Mindlin, Marlene Kretschmer"
copyright = f"{datetime.now():%Y}, Leipzig Institute for Meteorology (LIM)"

try:
    from importlib.metadata import version as _version

    release = _version("storypy")
except Exception:  # building from a checkout that is not installed
    release = ""
version = ".".join(release.split(".")[:2])

# --- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "myst_parser",          # Markdown pages (CHANGELOG.md, CONTRIBUTING.md)
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_timeout = 30

# Only ESMValTool is mocked: it is conda-only, and storypy imports it lazily.
# Everything else is installed on Read the Docs (see .readthedocs.yaml), so the
# API pages show real signatures and defaults. Mocking numpy/xarray used to make
# the plotting modules fail to import, leaving their pages empty.
autodoc_mock_imports = ["esmvaltool"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autoclass_content = "class"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = []

# --- HTML output --------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = f"storypy {release}" if release else "storypy"
html_static_path = []
# Copied to the site root: logos and the Google Search Console verification file.
html_extra_path = ["images", "google0b2594d760cfdda3.html"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "sticky_navigation": True,
}
html_context = {
    "display_github": True,
    "github_user": "LIM-Climate-Causality",
    "github_repo": "storypy",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
