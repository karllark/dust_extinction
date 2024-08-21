---
title: 'dust_extinction: Interstellar Dust Extinction Models'
tags:
  - Python
  - astronomy
  - milky way
  - magellanic clouds
  - dust
  - extinction
  - interstellar
authors:
  - name: Karl D. Gordon
    orcid: 0000-0001-5340-6774
    affiliation: 1
affiliations:
 - name: Space Telescope Science Institute, 3700 San Martin Drive, Baltimore, MD, 21218, USA
   index: 1
date: 22 May 2024
bibliography: paper.bib
---

# Summary

Extinction describes the effects of dust on observations of a single star due
to the dust along the line-of-sight to the star removing flux by absorbing
photons and scattering photons out of the line-of-sight. The wavelength
dependence of dust extinction (also known as extinction curves) provides
fundamental information about the size, composition, and shape of interstellar
dust grains. In general, models giving the wavelength dependence of extinction
are used to model or correct the effects of dust on observations. This Python
Astropy-affiliated package [@Astropy22] provides many of the published
extinction models in one place with a consistent interface.

# Statement of need

Many observation- and theory-based extinction curves have been presented in the
literature. Having one Python package providing these models ensures that they
are straightforward to use and used within their valid wavelength and parameter
(where appropriate) ranges. Other packages provide extinction curves, but they
generally provide one or a small number of curves for specialized purposes
[e.g., @Barbary16].

The types of extinction models supported are Averages, Parameter Averages,
Grain odels, and Shapes. The Averages are averages of a set of measured
extinction curves and examples are shown in \autoref{fig:averages}. The
Parameter Averages are extinction curve averages that depend on a parameter,
often $R(V) = A(V)/E(B-V)$ which is the ratio of total to selective extinction.
\autoref{fig:parameter_averages} shows examples of such models. The Grain odels
are those extinction curves computed using dust grain models
(\autoref{fig:grain}). Note that only these models provide dust extinction
predictions from X-ray through submm wavelengths. The final type of models are
Shapes that provide flexible functional forms that fit selected wavelength
ranges (see \autoref{fig:shapes} for examples).

![Examples of Average models based on observations in the Milky Way, Large Magellanic Cloud (LMC), and Small Magellanic Cloud (SMC) [@Bastiaansen92; @Gordon03; @Gordon09; @Gordon21; @Gordon24].\label{fig:averages}](average_models_uv_nir.png){
width=70% }

![Examples of Parameter Average models [@Cardelli89; @ODonnell94; @Fitzpatrick99; @Fitzpatrick04; @Valencic04; @Gordon09; @MaizApellaniz14; @Fitzpatrick19; @Decleir22; @Gordon23].\label{fig:parameter_averages}](parameter_average_models.png)

![Examples of Grain models that are based on fitting observed extinction curves as well as other dust observables (e.g., emission and polarization) [@Desert90; @Weingartner01; @Draine03; @Zubko04; @Compiegne11; @Jones13; @Hensley23; @Ysard24].\label{fig:grain}](grain_models.png){ width=70% }

![Example of a Shape model that is focused on decomposing the UV extinction curve [@Fitzpatrick90].\label{fig:shapes}](shape_models.png){ width=70% }

The wavelength dependence of extinction for a model is computed by passing a
wavelength or frequency vector with units. Each model has a valid wavelength
range which is enforced, as extrapolation is not supported. The model output is
in the standard $A(\lambda)/A(V)$ units where $A(\lambda)$ is the extinction in
magnitudes at wavelength $\lambda$ and $A(V)$ is the extinction in magnitudes
in the V band. Every model has a helper `extinguish` function that
alternatively provides the fractional effects of extinction for a specific dust
column (e.g., $A(V)$ value). This allows for the effects of dust to be modeled
for or removed from an observation.

This package does not implement dust attenuation models[^1]. Dust attenuation
results when observing more complex systems such as a star with nearby,
circumstellar dust or a galaxy with many stars extinguished by different
amounts of dust. In both cases, the wavelength dependence of effects of dust
are dependent not just on the dust grain properties, but also the effects of
the dust radiative transfer [@Steinacker13]. Specifically, these effects are
the averaging of sources extinguished by differing amount of dust and the
inclusion of a significant number of photons scattered into the observing beam.

[^1]: See [karllark/dust_attenuation](https://github.com/karllark/dust_attenuation).

Any published dust extinction model is welcome for inclusion in this package. I
thank Kristen Larson and Aidan McBride for model contributions and P. L. Lim,
Brigitta Sipocz, Soham Gaikwad, Clement Robert, and Adam Ginsburg for smaller
much appreciated contributions.

# References
