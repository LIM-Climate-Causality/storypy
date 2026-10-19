*********************
storypy documentation
*********************

Sources for https://storypy.readthedocs.io (Sphinx, Read the Docs theme).

Build locally
=============

From the repository root::

    pip install -e ".[docs]"
    make -C docs html
    open docs/build/html/index.html

To fail on any warning, as the CI docs job does::

    sphinx-build -W --keep-going -b html docs/source docs/build/html

Layout
======

- ``source/index.rst``: landing page and table of contents.
- ``source/getting_started/``: overview, installation, quickstart, configuration, piControl recipe.
- ``source/reference/``: API reference, generated from the docstrings.
- ``source/development/``: contributing guide and changelog (included from the repository's
  ``CONTRIBUTING.md`` and ``CHANGELOG.md``).

Read the Docs builds from ``.readthedocs.yaml`` at the repository root.
