.. _overview:

.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />

Overview
========

**StoryPy aim to facilitate analyzing dynamical storylines by providing efficient and user-friendly tools that is flexible and adaptable for various storyline research and policy applications..**

Representing model uncertainty may be relevant for a lot of applications, yet the analysis is not trivial and requires a lot of data processing and expert knowledge. This tool can help bring together experts with stakeholders to understand uncertainty at the regional scale with the technical part not being an issue, leaving more time for science, interpretation and communication.


What are storylines?
--------------------

The uncertainty in the response of the climate system to anthropogenic forcing is large, at regional scales, this uncertainty is associated with uncertain atmospheric circulation, such as the position of the storm track, the frequency of weather regimes, or the change in ENSO-driven teleconnections.

Dynamical storylines explore plausible changes in regional climate driven by qualitatively different (yet plausible) forced responses in large-scale remote drivers, such as polar amplification, tropical amplification, and the stratospheric polar vortex. In this way, storylines use physical understanding to link large-scale thermodynamic and dynamic climate responses to regional impacts and present a small set of projections in a conditional way.

It is said that a forced response is plausible when a global climate model projects such a change, this is why storylines are evaluated leveraging differences in ensembles of Global Circulation Models (GCMs) contributing to the Coupled Model Intercomparison Project (CMIP). This approach, as proposed by Zappa&Shepherd, 2017, helps address uncertainties in regional climate responses.

- The multimodel mean and the treatment of the large uncertainty around it in probabilistic terms is often not really meaningful for decision-making.

- Dynamical storylines provide a physically grounded framework to interpret the spread in the models by linking regional responses to variations in large-scale circulation drivers.

Methodology
-----------

Following the pattern scaling assumption described in Tebaldi & Arblaster, 2014, the end-of-century climate change response :math:`∆C_{xm}` in a field :math:`C` at location :math:`x`, in model :math:`m`, is expressed as a linear function of global warming :math:`∆T_m` and the climate response pattern :math:`P_{xm}`

.. math::

   \Delta C_{xm} = \Delta T_m P_{xm}

Pattern response (:math:`∆P_{xm}` at location :math:`x` and model :math:`m` proposed in Zappa&Shepherd, 2017), and also adopted in other storyline studies (e.g. Mindlin et al. 2020, Ghosh et al. 2023, Monerie et al. 2023) is used to quantify the influence of multiple sources of uncertainty, expressed as a linear combination of the response of the remote drivers scaled by global warming :math:`∆T`.

.. math::

   P_{xm} = a_x
          + b_x \left(\frac{\Delta T_{driver1}}{\Delta T}\right)'_m
          + c_x \left(\frac{\Delta T_{driver2}}{\Delta T}\right)'_m
          + d_x \left(\frac{\Delta T_{driver3}}{\Delta T}\right)'_m
          + e_{xm}


What is StoryPy?
---------------

**A user-friendly toolkit for analyzing dynamical storylines…**

StoryPy implements the dynamical storyline framework using CMIP model output. It provides

- a set of functions to analyze multi-model ensembles by focusing on the identification of dynamical storylines.

- customizable options for selecting remote drivers :math:`X`, target seasons, and climate variables or climatic-impact drivers :math:`C_x`.

We designed two options for processing CMIP data:

1. Option A: Using ESMValTool (via ESMValTool recipes) to download and preprocess the CMIP datasets, including regridding.

  Requirements:
    - ESMValTool installation and a working ESMValTool environment.
    - ESMValTool recipes / configuration compatible with target variables and drivers.
    - ESMValTool preprocessing can generate large intermediate files. Ensure sufficient disk space in the working directory.

2. Option B: Using a local CMIP database where StoryPy reads CMIP-style NetCDF files directly from a local directory. Provided that:
    - CMIP datasets already available locally (or accessible via a mounted filesystem).
    - It follows the naming and grid conventions as described below.
      
    >>> <data_dir>/
    >>> ├── <var_name>/
    >>> │   ├── <mon>/
    >>> │   │   ├── <g025>/
    >>> │   │   │   ├── <var_name>_<period>_<model>_<experiment>_<member>_<grid>.nc
    >>> │   │   │   └── ...
    >>> │   │   └── ...
    >>> │   └── ...
    >>> └── ...

Preprocessing CMIP data is a crucial step in the analysis of dynamical storylines, as it ensures that the data is in a consistent format and resolution for analysis. StoryPy provides two options for preprocessing CMIP data (as already described), either using ESMValTool or by reading from a local CMIP database. The choice between these options depends on the user's preferences and the availability of data. Given CMIP data, user can preprocess the data by calling the methods and using the following steps for example:

>>> from storypy.preprocess import ESMValProcessor, ModelDataPreprocessor, parse_config
>>> processor_target = ESMValProcessor(esmval_config, user_config, driver_config)
OR
>>> processor_target = ModelDataPreprocessor(user_config, driver_config)
>>> processor_target.process_var()
>>> processor_target.process_driver()

Users can compute the driver indices and regression coefficients for a desired study region, for example:

For the driver indices:

>>> from storypy.compute import compute_drivers
>>> df_raw, df_scaled, df_standardized = compute_drivers(driver_config)

For the regression coefficients:

>>> from storypy.compute import run_regression
>>> outputs = run_regression(user_config)

What storypy does not do (Limitations)
--------------------------------------

After motivating you on the advantages of using storypy, we also want to bring to your attention what storypy currently does not do. Storypy is designed to make it easier to build and compare storylines from climate-model output, but there are important cases where StoryPy cannot guarantee “correct” or comparable results. In particular, StoryPy assumes that the inputs it receives are physically meaningful, consistently processed, and comparable across models. The following limitations are worth keeping in mind.

1. **Regridded data and other processing irregularities**

Storypy can work with data that has been regridded or post-processed, but it does not automatically detect or correct inconsistencies introduced upstream. Small differences in preprocessing can translate into noticeable differences in indices, scaling factors, regression coefficients, or storyline patterns. As an example, we preprocessed CMIP6 data using ESMValTool and compared the results to a local CMIP database that we had previously processed. We found that the results were not exactly the same, even though the same models and variables were used. This is because of differences in regridding methods, interpolation, and other processing steps. Therefore, it is important to ensure that the data is processed consistently across models and that any differences in preprocessing are understood and accounted for when interpreting the results. |brr|

.. image:: images/fig_pr.png
   :width: 80%
   :alt: PR changes
   :align: center

2. **Fundamental problems with model data (“garbage in, garbage out”)**

Storypy cannot guarantee that a given climate model dataset is suitable for the question you want to answer. Like any analysis tool, Storypy will faithfully process the inputs it is given, even if the underlying data contain biases or structural problems that no post-processing step can fix.

.. figure:: images/fig_pr.png
   :width: 800
   :align: center
   :alt: PR changes

   This figure shows the changes in PR over time.

About the authors
-----------------

**Richard** is a PhD student of Jun. Prof. Marlene Kretschmer at the Leipzig Institutate for Meteorology, Leipzig University. His research interests lie at the intersection of atmospheric science, climate modeling, and data science, aiming to tackle pressing global challenges like climate change. Richard holds an MSc in Environmental Physics from the University of Bremen (Germany), and a BTech in Physics Electronics from the Federal university of Technology, Minna (Nigeria).

**Julia** is a postdoctoral researcher at the University of Leipzig, working with Jun. Prof. Marlene Kretschmer. She is interested in how large scale variability and change can influence regional climate. In particular, she is interested in South America because it is the region where she grew up. However, this initial interest has led to a general interest in large scale circulation dynamics of the Southern Hemisphere and its remote drivers, such as tropical modes of variability such as El Nino Southern Oscillation and the Indian Ocean Dipole and the stratosphere.

**Marlene** studied mathematics before completing a PhD in climate physics at the Potsdam Institute for Climate Impact Research. She then worked as a postdoctoral researcher in the Department of Meteorology at the University of Reading (UK). Since 2022, She has been a Junior Professor of Climate Causality at Leipzig University (Germany).

Get in touch
------------

If you have suggestions on additional methods we could add, questions you'd like to ask, issues that you are finding in the application of the methods that are already implemented, or bugs in the code, please contact us under ...@gmail.com or `raise an issue on github <https://github.com/LIM-Climate-Causality/storypy/issues>`_.

Cite the package
----------------

If you use storypy for your research, please cite our publication:

Alawode, R., Mindlin, J., Kretschmer, M.: ...

References
----------

- Ghosh, R., & Shepherd, T. G. (2023). Storylines of Maritime Continent dry period precipitation changes under global warming. Environmental Research Letters, 18(3), 034017. https://doi.org/10.1088/1748-9326/acb788.
- Levine, X. J., Williams, R. S., Marshall, G., Orr, A., Graff, L. S., Handorf, D., Karpechko, A., Köhler, R., Wijngaard, R. R., Johnston, N., Lee, H., Nieradzik, L., & Mooney, P. A. (2024). Storylines of summer Arctic climate change constrained by Barents–Kara seas and Arctic tropospheric warming for climate risk assessment. Earth System Dynamics, 15(4), 1161–1177. https://doi.org/10.5194/esd-15-1161-2024.
- Mindlin, J., Shepherd, T. G., Vera, C. S., Osman, M., Zappa, G., Lee, R. W., & Hodges, K. I. (2020). Storyline description of Southern Hemisphere midlatitude circulation and precipitation response to greenhouse gas forcing. Climate Dynamics, 54(9–10), 4399–4421. https://doi.org/10.1007/s00382-020-05234-1.
- Monerie, P., Biasutti, M., Mignot, J., Mohino, E., Pohl, B., & Zappa, G. (2023). Storylines of Sahel Precipitation change: Roles of the North Atlantic and Euro‐Mediterranean temperature. Journal of Geophysical Research Atmospheres, 128(16). https://doi.org/10.1029/2023jd038712.
- Tebaldi, C., & Arblaster, J. M. (2014). Pattern scaling: Its strengths and limitations, and an update on the latest model simulations. Climatic Change, 122(3), 459–471. https://doi.org/10.1007/s10584-013-1032-9.
- Zappa, G., & Shepherd, T. G. (2017). Storylines of atmospheric circulation change for European Regional Climate Impact Assessment. Journal of Climate, 30(16), 6561–6577. https://doi.org/10.1175/jcli-d-16-0807.1.
