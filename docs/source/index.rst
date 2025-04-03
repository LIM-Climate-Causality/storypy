.. ibicus documentation master file, created by
   sphinx-quickstart on Wed Mar 30 16:04:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to storypy's documentation!
========================================

**storypy is a python based packages that provides a tailored interface for computing climate storylines.**

storypy privides:

- a set of functions to analyze multi‐model ensembles by focusing on the identification of dynamical storylines.

- customizable options for selecting remote drivers, target seasons, and climate variables or climatic‐impact drivers, the storypy provides flexibility and adaptability for various research and policy applications

How to cite: Alawode, R., Mindlin, J., Kretschmer, M........, 2024.

Documentation
_____________

**Getting Started**

* :doc:`getting_started/overview`
* :doc:`getting_started/installing`
* :doc:`getting_started/whatarestorylines`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   getting_started/overview
   getting_started/installing
   getting_started/whatarestorylines

**Tutorials**

- `Zappa_and_Shepherd_2017 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/zappa_shepherd.ipynb>`_
- `Mindlin_2020 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/mindlin_2020>`_
- `Monerie_2023 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/monerie_2023.ipynb>`_
- `Ghosh_2023 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/ghosh_2023.ipynb>`_


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   Zappa_and_Shepherd_2017 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/zappa_shepherd.ipynb>
   Mindlin_2020 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/mindlin_2020>
   Monerie_2023 Debiasers <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/monerie_2023.ipynb>
   Ghosh_2023 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/ghosh_2023.ipynb>

**Documentation / API reference**

* :doc:`reference/api`
   * :doc:`reference/debias`
   * :doc:`reference/evaluate`
   * :doc:`reference/utils`
   * :doc:`reference/variables`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   ibicus.debias module <reference/debias>
   ibicus.evaluate module <reference/evaluate>
   ibicus.utils module <reference/utils>
   ibicus.variables module <reference/variables>

License
-------

storypy is available under the open source `MIT License`__.

__ https://github.com/LIM-Climate-Causality/storypy/blob/main/LICENSE


Acknowledgements
----------------

The development of this package was supported by the European Centre for Mid-term Weather Forecasts (ECMWF) as part of the `ECMWF Summer of Weather Code <https://esowc.ecmwf.int/>`_

.. image:: images/logos.png
   :width: 800
   :alt: ECMWF logos
