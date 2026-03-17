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

**Methodology**

Following the pattern scaling assumption described in Tebaldi & Arblaster, 2014, the end-of-century climate change response :math:`∆C_{xm}` in a field :math:`C` at location :math:`x`, in model :math:`m`, is expressed as a linear function of global warming :math:`∆T_m` and the climate response pattern :math:`P_{xm}`

.. math::

   \Delta C_{xm} = \Delta T_m P_{xm}

Pattern response (:math:`∆P_{xm}` at location :math:`x` and model :math:`m` proposed in Zappa&Shepherd, 2017), and also adopted in other storyline studies (e.g. Mindlin et al. 2020, Ghosh et al. 2020, Monerie et al. 2021) is used to quantify the influence of multiple sources of uncertainty, expressed as a linear combination of the response of the remote drivers scaled by global warming.

.. math::

   P_{xm} = a_x
          + b_x \left(\frac{\Delta T_{driver1}}{\Delta T}\right)'_m
          + c_x \left(\frac{\Delta T_{driver2}}{\Delta T}\right)'_m
          + d_x \left(\frac{\Delta T_{driver3}}{\Delta T}\right)'_m
          + e_{xm}


What is StoryPy?
---------------

**A user-friendly toolkit to analyze dynamical storylines…**

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
>>> outputs = run_regression(main_config)

What storypy cannot guarantee
-----------------------------

After motivating you on the advantages of using storypy, we also want to bring to your attention what storypy currently does not do:

1. ibicus offers a multivariate evaluation of the bias adjusted climate model but does not currently support multivariate bias adjustment, meaning the correction of spatial or inter-variable structure. Whether or not to correct for example the inter-variable structure, which could be seen as an integral feature of the climate model, is a contentious and debated topic of research. If such correction is necessary, the excellent `MBC <https://cran.r-project.org/web/packages/MBC/index.html>`_ or `SBCK <https://github.com/yrobink/SBCK>`_ package are suitable solutions. For a more detailed discussion of the advantages and possible drawbacks of multivariate bias adjustment we refer to Spuler et al. (2023) cited above. |brr|

2. ibicus is not suitable for 'downscaling' the climate model which is a term for methods used to increase the spatial resolution of climate models. Although bias adjustment methods have been used for downscaling, in general they are not appropriate, since they do not reproduce the local scale variability that is crucial on those scales. Maraun 2016 argues that for downscaling, stochastic methods have great advantages. An example of a package addressing the problem of downscaling is: `Rglimclim <https://www.ucl.ac.uk/~ucakarc/work/glimclim.html>`_. |brr|

3. 'Garbage in, garbage out'. ibicus cannot guarantee that the climate model is suitable for the problem at hand. As mentioned above, although bias adjustment can help with misspecifications, it cannot solve fundamental problems within climate models. The evaluation framework can help you identify whether such fundamental issues exist in the chosen climate model. However, this cannot replace careful climate model selection before starting a climate impact study. |brr|

About the authors
-----------------

Richard is a PhD student of Jun. Prof. Marlene Kretschmer at the Leipzig Institutate for Meteorology, Leipzig University. His research interests lie at the intersection of atmospheric science, climate modeling, and data science, aiming to tackle pressing global challenges like climate change. Richard holds an MSc in Environmental Physics from the University of Bremen (Germany), and a BTech in Physics Electronics from the Federal university of Technology, Minna (Nigeria).

Julia is a postdoctoral researcher at the University of Leipzig, working with Jun. Prof. Marlene Kretschmer. She is interested in how large scale variability and change can influence regional climate. In particular, she is interested in South America because it is the region where she grew up. However, this initial interest has led to a general interest in large scale circulation dynamics of the Southern Hemisphere and its remote drivers, such as tropical modes of variability such as El Nino Southern Oscillation and the Indian Ocean Dipole and the stratosphere.

Marlene studied mathematics before completing a PhD in climate physics at the Potsdam Institute for Climate Impact Research. She then worked as a postdoctoral researcher in the Department of Meteorology at the University of Reading (UK). Since 2022, She has been a Junior Professor of Climate Causality at Leipzig University.

Get in touch
------------

If you have suggestions on additional methods we could add, questions you'd like to ask, issues that you are finding in the application of the methods that are already implemented, or bugs in the code, please contact us under ...@gmail.com or `raise an issue on github <https://github.com/LIM-Climate-Causality/storypy/issues>`_.


Cite the package
----------------

If you use ibicus for your research, please cite our publication in Geoscientific Model Development:

Spuler, F. R., Wessel, J. B., Comyn-Platt, E., Varndell, J., and Cagnazzo, C.: ibicus: a new open-source Python package and comprehensive interface for statistical bias adjustment and evaluation in climate modelling (v1.0.1), Geosci. Model Dev., 17, 1249–1269, https://doi.org/10.5194/gmd-17-1249-2024, 2024.

References
----------

- Maraun, D. Bias Correcting Climate Change Simulations - a Critical Review. Curr Clim Change Rep 2, 211–220 (2016). https://doi.org/10.1007/s40641-016-0050-x
- Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? In Journal of Climate (Vol. 28, Issue 17, pp. 6938–6959). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00754.1
- Switanek, M. B., Troch, P. A., Castro, C. L., Leuprecht, A., Chang, H.-I., Mukherjee, R., & Demaria, E. M. C. (2017). Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes. In Hydrology and Earth System Sciences (Vol. 21, Issue 6, pp. 2649–2666). Copernicus GmbH. https://doi.org/10.5194/hess-21-2649-2017.
- Michelangeli, P.-A., Vrac, M., & Loukos, H. (2009). Probabilistic downscaling approaches: Application to wind cumulative distribution functions. In Geophysical Research Letters (Vol. 36, Issue 11). American Geophysical Union (AGU). https://doi.org/10.1029/2009gl038401
- Famien, A. M., Janicot, S., Ochou, A. D., Vrac, M., Defrance, D., Sultan, B., & Noël, T. (2018). A bias-corrected CMIP5 dataset for Africa using the CDF-t method – a contribution to agricultural impact studies. In Earth System Dynamics (Vol. 9, Issue 1, pp. 313–338). Copernicus GmbH. https://doi.org/10.5194/esd-9-313-2018
- Vrac, M., Drobinski, P., Merlo, A., Herrmann, M., Lavaysse, C., Li, L., & Somot, S. (2012). Dynamical and statistical downscaling of the French Mediterranean climate: uncertainty assessment. In Natural Hazards and Earth System Sciences (Vol. 12, Issue 9, pp. 2769–2784). Copernicus GmbH. https://doi.org/10.5194/nhess-12-2769-2012
- Vrac, M., Noël, T., & Vautard, R. (2016). Bias correction of precipitation through Singularity Stochastic Removal: Because occurrences matter. In Journal of Geophysical Research: Atmospheres (Vol. 121, Issue 10, pp. 5237–5258). American Geophysical Union (AGU). https://doi.org/10.1002/2015jd024511
- Li, H., Sheffield, J., and Wood, E. F. (2010), Bias correction of monthly precipitation and temperature fields from Intergovernmental Panel on Climate Change AR4 models using equidistant quantile matching, J. Geophys. Res., 115, D10101, doi:10.1029/2009JD012882.
- Lange, S. (2019). Trend-preserving bias adjustment and statistical downscaling with ISIMIP3BASD (v1.0). In Geoscientific Model Development (Vol. 12, Issue 7, pp. 3055–3070). Copernicus GmbH. https://doi.org/10.5194/gmd-12-3055-2019
- Lange, S. (2022). ISIMIP3BASD (3.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.6758997
