.. storypy documentation master file

Welcome to storypy's documentation!
===================================

**storypy** is a Python package that provides a tailored interface for computing climate storylines.

It offers:
----------

- A set of functions to analyze multi‐model ensembles, focusing on the identification of dynamical storylines.
- Customizable options for selecting remote drivers, target seasons, and climate variables or climatic‐impact drivers.
- Flexibility and adaptability for various research and policy applications.

How to cite:
------------

Alawode, R., Mindlin, J., Kretschmer, M., *et al.* (2024).  
*storypy: A Python interface for climate storyline computation.*

---

Getting Started
===============

These pages will help you install and use **storypy** for the first time.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/overview
   getting_started/installing
   getting_started/whatarestorylines

---

Tutorials
=========

Below are example workflows that demonstrate how to use **storypy** in practice.  
Each notebook is hosted on **nbviewer** and linked directly.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   Zappa_and_Shepherd_2017 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/zappa_shepherd.ipynb>
   Mindlin_2020 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/mindlin_2020>
   Monerie_2023 Debiasers <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/monerie_2023.ipynb>
   Ghosh_2023 <https://nbviewer.org/github/LIM-Climate-Causality/storypy/blob/main/notebooks/ghosh_2023.ipynb>

---

API Reference
=============

This section provides the technical documentation for **storypy**, including all modules, classes, and functions.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/api
   .. reference/preprocess
   .. reference/compute
   .. reference/evaluate
   .. reference/data
   .. reference/utils

---

License
=======

**storypy** is available under the open source `MIT License`__.

__ https://github.com/LIM-Climate-Causality/storypy/blob/main/LICENSE

---

Acknowledgements
================

The development of this package was supported by the **Leipzig Institute for Meteorology (LIM)**,  
with partnership funding from the **Deutsche Forschungsgemeinschaft (DFG)** under AC3.

.. image:: images/logos.png
   :width: 800
   :alt: ECMWF logos
   :align: center

