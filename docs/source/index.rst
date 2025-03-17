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

The ibicus documentation presented here provides a detailed overview of the different methods implemented, their default settings and possible modifications in parameters under `Documentation - ibicus.debias <reference/debias.html>`_, as well as a detailed description of the evaluation framework under `Documentation - ibicus.evaluate <reference/debias.html>`_ For a hands-on introduction to the package see our tutorial notebooks.

The documentation also provides a brief introduction to bias adjustment and possible issues with the approach under `Getting started <getting_started>`_. For a more detailed introduction to bias adjustment, as well as an overview of relevant literature on existing methods and issues, we refer to our paper published in Geoscientific Model Development:

How to cite: Alawode, R., Mindlin, J., Kretschmer, M........, 2024.

Documentation
_____________

**Getting Started**

* :doc:`getting_started/overview`
* :doc:`getting_started/installing`
* :doc:`getting_started/whatisdebiasing`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   getting_started/overview
   getting_started/installing
   getting_started/whatisdebiasing

**Tutorials**

- `Zappa_and_Shepherd_2017 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/zappa_shepherd.ipynb>`_
- `Mindlin_2020 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/mindlin_2020>`_
- `Monerie_2023 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/monerie_2023.ipynb>`_
- `Ghosh_2023 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/ghosh_2023.ipynb>`_
- `04 Parallelization and Advanced Topics <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/04%20Parallelization%20and%20Advanced%20Topics.ipynb>`_


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   00 Download and Preprocess Data <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/00%20Download%20and%20Preprocess.ipynb>
   01 Getting Started <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/01%20Getting%20Started.ipynb>
   02 Adjusting Debiasers <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/02%20Adjusting%20Debiasers.ipynb>
   03 Evaluation <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/03%20Evaluation.ipynb>
   04 Parallelization and Advanced Topics <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/04%20Parallelization%20and%20Advanced%20Topics.ipynb>

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

ibicus is available under the open source `Apache-2.0 License`__.

__ https://github.com/ecmwf-projects/ibicus/blob/main/LICENSE


Acknowledgements
----------------

The development of this package was supported by the European Centre for Mid-term Weather Forecasts (ECMWF) as part of the `ECMWF Summer of Weather Code <https://esowc.ecmwf.int/>`_

.. image:: images/logos.png
   :width: 800
   :alt: ECMWF logos
