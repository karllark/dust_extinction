---
title: 'dust_extinction: Interstellar Dust Extinction Models'
tags:
  - Python
  - astronomy
  - milky way
authors:
  - name: Karl D. Gordon
    orcid: 0000-0001-5340-6774
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Space Telescope Science Institute, 3700 San Martin Drive, Baltimore, MD, 21218, USA
   index: 1
date: 21 May 2024
bibliography: paper.bib
---

# Summary

Extinction describes the effects of dust on observations of single star. The
dust along the line-of-sight to a star removes flux by absorbing photons and
scattering photons out of the line-of-sight. The wavelength dependence of dust
extinction (also know as extinction curves) provides fundamental information
about the size, composition, and shape of interstellar dust grain. In general,
models giving the wavelength dependence of extinction are used to model or
correct the effects of dust on observations a single star.  This python package
provides most of the extinction models published in one place with a 
straightforward interface.

# Statement of need

Many observational and theoretical based extinction curves have been presented
in the literature. Having one python package providing these models ensures
that they are straightforward to use and are used within their valid wavelength
ranges.

The types of extinction models supported are Averages, Parameter Averages, Grain Models, and Shape.  [explain each kind]  [provide plots]

![Parameter Average models [@Gordon23].\label{fig:parameter_averages}](parameter_average_models.png)

The wavelength dependence of extinction for a model is computing by passing a
wavelength or frequency vector with astropy units. Each model has a valid
wavelength range this is enforced to ensure as extrapolation is not supported.
The model output is in the standard $A(\lambda)/A(V)$ units where $A(\lambda)
is the extinction at wavelength $\lambda$ and $A(V)$ is the extinction in the 
Johnson V band.  Every model has a helper `extinguish` function that alternatively
provides the fractional effects of extinction for a specific dust column $A(V)$.
This allows for the effects of dust to be modeled for or removed from an observation.

This package does not implement dust attenuation models. Dust attenuation
results in observations of more complex systems like a star with nearby,
circumstellar dust or a galaxy with many stars extinguished by different
amounts of dust. In both cases, a significant fraction of the observed photons
come from scattering into the observer beam. Thus, the wavelength dependence of
dust attenuation is dependent on the dust grain properties and dust radiative
transfer.

# References