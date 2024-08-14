##################
Base Model Classes
##################

All the extinction models are based on either provided in dust_extinction
or an astropy.modeling `FittableModel`.  

For examples of how all the classes in ``dust_extinction`` are used, see the
implemented models.

.. _allmods:
All
===

All of these classes used in ``dust_extinction`` are based on the 
`Model <https://docs.astropy.org/en/stable/modeling/>`_ astropy.modeling class.
See the astropy docs for the full details of all that is possible with this class.

All dust extinction models have the following:

* A member variable `x_range` that that define the valid range of wavelengths.
These are defined in inverse microns for historical reasons.
* A member function `evaluate` that computes the extinction at a given `x` and
any model parameter values.  The `x` values are checked to be within the valid `x_range`.
The `x` values should have astropy.units.  If they do not, then they are assumed
to be in inverse microns and a warning is issued stating such.

FittableModel
=============

The ``dust_extinction`` shape models are based on the astropy `FittableModel`. 
One general use case for these models is to fit observed dust extinction curves.
See :ref:`fit_curve`.  These models follow the standard astropy setup for such
models.  This includes defining the parameters to be fit with the `Parameter`
function.

Thus all `shape` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`)
* Member parameters that are defined with the astropy `Parameter` function.  
This includes default starting values and any standard bounds on the parameters.
The number and name of the paramaeters varies by `shape` model.
* The `evaluate` function that calculates the extinction curve based on the 
input parameters.

BaseExtModel
============

The :class:`~dust_extinction.BaseExtModel` provides the base model for all 
the rest of the `dust_extinction` models.   This model provides the 
`extinguish` member function (see :ref:`extinguish_example`).

All of the `average` models are based on `BaseExtModel` directly.  Thus 
all the `average` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`).
The `evaluate` function may interpolate the observed average extinction curve or 
it may be based on a `shape` fit to the observed data.
* The member function `extinguish`.
* A member parameter `Rv` that gives the ratio of absolute to selective extinction
(i.e., R(V) = A(V)/E(B-V)).  This is not set with the astropy `Parameter` function 
as is included for reference.
* Member variables that give the tabulated observed extinction curve as a function
of wavelength.  The variables for this information are `obsdata_x` and `obsdata_axav`.
The accuracy of this tabulated information is given as `obsdata_tolerance` and this
is used for the automated testing and in plotting.
Some models also have an `obsdata_azav_unc` if such is available from the literature.

BaseExtRvModel
==============

The :class:`~dust_extinction.BaseExtRvModel` provides the base model for all 
the ``dust_extinction`` models that are depending on `Rv` only.  `Rv` is the
ratio of absolute to selective extinction (i.e., R(V) = A(V)/E(B-V)).

These are the majority of the `parameter_average` models and they have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`).
The `evaluate` function that calculates the extinction curve based on the 
`Rv` value.
* The member function `extinguish`.
* A member variable `Rv` defined using the astropy `Parameter` function.
* A member variable `Rv_range` that provides the valid range of `Rv` values.
* A validator member function called `Rv` tagged with `@Rv.validator` that validates
the input `Rv` based on the `Rv_range`.

BaseExtRvfAModel
================

The :class:`~dust_extinction.BaseExtRvfAModel` provides the base model for all 
the ``dust_extinction`` models that are depending on `Rv` and `fA`.

These `parameter_average` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`).
The `evaluate` function that calculates the extinction curve based on the 
`Rv` and `fA` values.
* The member function `extinguish`.
* Member variables `Rv` and `fA` defined using the astropy `Parameter` function.
* A member variable `Rv_range` that provides the valid range of `Rv` values.
* A member variable `fA_range` that provides the valid range of `fA` values.
* A validator member function called `Rv` tagged with `@Rv.validator` that validates
the input `Rv` based on the `Rv_range`.
* A validator member function called `fA` tagged with `@fA.validator` that validates
the input `fA` based on the `fA_range`.

BaseExtGrainModel
=================

The :class:`~dust_extinction.BaseExtGrainModel` provides the base model for all 
the ``dust_extinction`` models that are based on dust grain models.  All these 
models are provided as tabulated data tables.

These `grain_model` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`).
The `evaluate` function thats interpolates the model extinction curve.
* The member function `extinguish`.
* A member parameter `possnames` that is a dictonary with a key that is a tag for the
model (e.g., `MWRV31`) and a tuple that is (filename, Rv).  This key is used when 
initialized a `grain_model`.