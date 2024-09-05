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

All dust extinction models have at least the following:

* A member variable `x_range` that that define the valid range of wavelengths. These are defined in inverse microns as is common for extinction curve research.
* A member function `evaluate` that computes the extinction at a given `x` and any model parameter values.  The `x` values are checked to be within the valid `x_range`. The `x` values passed to the `evaluate` method have no units; the base class `BaseExtModel` will automatically convert whatever units the user provided to inverse microns prior to calling the `evaulate` method. The `evaluate` method should not be called directly.

All of these classes used in ``dust_extinction`` are based on the 
`Model <https://docs.astropy.org/en/stable/modeling/>`_ astropy.modeling class.
See the astropy docs for the full details of all that is possible with this class.

FittableModel
=============

The ``dust_extinction`` `shape` models are based on the astropy `FittableModel`. 
One general use case for these models is to fit observed dust extinction curves.
See :ref:`fit_curves`.  These models follow the standard astropy setup for such
models.  This includes defining the parameters to be fit with the astropy `Parameter`
function.

Thus all `shape` models have:

* The member variable `x_range` and function `evaluate` are set in each `shape` model explicitly.
* Member parameters are explicitly defined with the astropy `Parameter` function.  This includes default starting values and any standard bounds on the parameters. The number and name of the paramaeters varies by `shape` model.
* The `evaluate` function that calculates the extinction curve based on the input parameters.  This is explicitly defined in each `shape` model.

BaseExtModel
============

The :class:`~dust_extinction.baseclasses.BaseExtModel` provides the base model for all 
the rest of the `dust_extinction` models.   This model provides the 
`extinguish` member function (see :ref:`extinguish_example`).

All of the `average` models are based on `BaseExtModel` directly.  Thus 
all the `average` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`). These are set explicitly for each `average` model.  The `evaluate` function may interpolate the observed average extinction curve or it may be based on a `shape` fit to the observed data.
* The member function `extinguish` inherited from the `BaseExtModel`.
* A member parameter `Rv` that gives the ratio of absolute to selective extinction (i.e., R(V) = A(V)/E(B-V)).  This is not set with the astropy `Parameter` function as is included mainly for reference.
* Member variables that give the tabulated observed extinction curve as a function of wavelength.  The variables for this information are `obsdata_x` and `obsdata_axav`. The accuracy of this tabulated information is given as `obsdata_tolerance` and this is used for the automated testing and in plotting. Some models also have an `obsdata_azav_unc` if such is available from the literature.

BaseExtRvModel
==============

The :class:`~dust_extinction.baseclasses.BaseExtRvModel` provides the base model for all 
the ``dust_extinction`` models that are depending on `Rv` only.  `Rv` is the
ratio of absolute to selective extinction (i.e., R(V) = A(V)/E(B-V)).  This model defines
the member variable `Rv` that is defined using the astropy `Parameter` function and a validator 
member function `Rv` that validates the input `Rv` is in the `Rv_range`.  This model is based 
on the `BaseExtModel`, hence inherits the `extinguish` member functionf

These are the majority of the `parameter_average` models and they have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`) are explicitly defined for each `paramter_average` model. The `evaluate` function calculates the extinction curve based on the `Rv` value.
* The member function `extinguish` inherited from the `BasedExtRvModel`.
* A member variable `Rv` inherited from the `BaseExtRvModel`.
* A member variable `Rv_range` that provides the valid range of `Rv` values.
* A validator member function called `Rv` tagged with `@Rv.validator` that validates the input `Rv` based on the `Rv_range`.  This is inherited from the `BaseExtRvModel`

BaseExtRvAfAModel
=================

The :class:`~dust_extinction.baseclasses.BaseExtRvAfAModel` provides the base model for all 
the ``dust_extinction`` models that are depending on `RvA` and `fA`.
These models are a mixture of two ``dust_extinction`` models where the A component
is dependent on `Rv` and the B component is not.
The `RvA` gives the R(V) value of component A and `fA` gives the fraction of the A 
component and (1 - fA) gives the fraction of the B component.
This model defines
the member variables `RvA` and `fA` that are defined using the astropy `Parameter` function and validator 
member functions `RvA` and `fA` that validate the input `RvA` and `fA` are in the `Rv_range` and `fA_range`. 
This model is based  on the `BaseExtModel`, hence inherits the `extinguish` member function.

These `parameter_average` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`). The `evaluate` function that calculates the extinction curve based on the `RvA` and `fA` values.
* The member function `extinguish` inherited from the `BasedExtRvAfAModel`.
* Member variables `RvA` and `fA` defined using the astropy `Parameter` function inherited from the `BasedExtRvAfAModel`.
* A member variable `RvA_range` that provides the valid range of `RvA` values inherited from the `BasedExtRvAfAModel`.
* A member variable `fA_range` that provides the valid range of `fA` values inherited from the `BasedExtRvAfAModel`.
* A validator member function called `RvA` tagged with `@RvA.validator` that validates the input `Rv` based on the `Rv_range` inherited from the `BasedExtRvAfAModel`.
* A validator member function called `fA` tagged with `@fA.validator` that validates the input `fA` based on the `fA_range` inherited from the `BasedExtRvAfAModel`.

BaseExtGrainModel
=================

The :class:`~dust_extinction.baseclasses.BaseExtGrainModel` provides the base model for all 
the ``dust_extinction`` models that are based on dust grain models.  All these 
models are provided as tabulated data tables.
This model defines a member function `evaluate` thats interpolates the model extinction curve.
This model is based  on the `BaseExtModel`, hence inherits the `extinguish` member function.

These `grain_model` models have:

* The member variable `x_range` and function `evaluate` (see :ref:`allmods`). The `evaluate` function is inherited from the `BaseExtGrainModel`.
* The member function `extinguish` inherited from the `BaseExtGrainModel`.
* A member parameter `possnames` that is a dictionary with a key that is a tag for the model (e.g., `MWRV31`) and a tuple that is (filename, Rv).  This key is used when initialized a `grain_model`.
* The member function `__init__` that reads in the tabular data into member functions `data_x` and `data_axav`.
