#####################
How to Choose a Model
#####################

The ``dust_extinction`` package provides a suite of dust extinction models.
Which model to use can depend on the wavelength range of interest, the expected
type of extinction, or some other property.

Average Models
==============

Simple Average Curves
---------------------

These are straightforward averages of observed extinction curves.  They are the
simplest models and include models for the MW
(:class:`~dust_extinction.averages.GCC09_MWAvg`), the LMC
(:class:`~dust_extinction.averages.G03_LMCAvg`,
:class:`~dust_extinction.averages.G03_LMC2`) and the SMC
(:class:`~dust_extinction.averages.G03_SMCBar`).

One often used alternative to these straight average models is to use one of
the parameter dependent models with the average R(V) value.  For the Milky
Way, the usual average used is R(V) = 3.1.

+--------------+-------------+------------------+--------------+
| Model        | x range     | wavelength range |       galaxy |
+==============+=============+==================+==============+
| GCC09_MWAvg  | 0.3 - 10.96 |     0.0912 - 3.3 |           MW |
+--------------+-------------+------------------+--------------+
| G03_LMCAvg   |  0.3 - 10.0 |        0.1 - 3.3 |          LMC |
+--------------+-------------+------------------+--------------+
| G03_LMC2     |  0.3 - 10.0 |        0.1 - 3.3 | LMC (30 Dor) |
+--------------+-------------+------------------+--------------+
| G03_SMCBar   |  0.3 - 10.0 |        0.1 - 3.3 |          SMC |
+--------------+-------------+------------------+--------------+


Parameter Dependent Average Curves
----------------------------------

The models that are dependent on parameters provide average curves that account
for overall changes in the extinction curve shapes.  For example, the average
behavior of Milky Way extinction curves has been shown to be dependent on R(V)
= A(V)/E(B-V).  R(V) roughly tracks with the average dust grain size.

The most general model is :class:`~dust_extinction.parameter_averages.G16` as this
model encompasses the average measured behavior of extinction curves in the MW,
LMC, and SMC.  The :class:`~dust_extinction.parameter_averages.G16` model reduces
to the :class:`~dust_extinction.parameter_averages.F99` model with f\ :sub:`A`\ =
1.0.  If only MW type extinction is expected, then the
:class:`~dust_extinction.parameter_averages.F04` model should be considered as it
is based on significantly more extinction curves than the
:class:`~dust_extinction.parameter_averages.CCM89` or
:class:`~dust_extinction.parameter_averages.O94` models.
The :class:`~dust_extinction.parameter_averages.M14` models focus on refining
models in the optical, and use the
:class:`~dust_extinction.parameter_averages.CCM89` models for the NIR and the UV.
The :class:`~dust_extinction.parameter_averages.M14` models use
R_5495 = A(5485)/E(4405-5495), the spectroscopic equivalent to
band-integrated R(V); see the paper for discussion.

+----------+-------------+-------------+------------------+--------------+
| Model    | Parameters  | x range     | wavelength range |       galaxy |
|          |             | [1/micron]  | [micron]         |              |
+==========+=============+=============+==================+==============+
| CCM89    |  R(V)       |  0.3 - 10.0 |        0.1 - 3.3 |           MW |
+----------+-------------+-------------+------------------+--------------+
| O94      |  R(V)       |  0.3 - 10.0 |        0.1 - 3.3 |           MW |
+----------+-------------+-------------+------------------+--------------+
| F99, F04 |  R(V)       |  0.3 - 10.0 |        0.1 - 3.3 |           MW |
+----------+-------------+-------------+------------------+--------------+
| M14      |  R_5495     |  0.3 - 10.0 |        0.1 - 3.3 |      MW, LMC |
+----------+-------------+-------------+------------------+--------------+
| G16      | R(V)_A, f_A |  0.3 - 10.0 |        0.1 - 3.3 | MW, LMC, SMC |
+----------+-------------+-------------+------------------+--------------+

Shape Models
============

The models that focus on describing the full extinction curve shape are usually
used to fit measured extinction curves.  These models allow features in the
extinction curve to be measured (e.g., 2175 A bump or 10 micron silicate
feature).  The :class:`~dust_extinction.shapes.P92` is the most
general as it covers the a very broad wavelength range.  The
:class:`~dust_extinction.shapes.FM90` model has been extensively used,
but only covers the UV wavelength range.

+------------+--------------+------------------+-------------------+
| Model      | x range      | wavelength range | # of parameters   |
+============+==============+==================+===================+
| FM90       | 3.13 - 11.0  |    0.0912 - 0.32 |  6                |
+------------+--------------+------------------+-------------------+
| P92        | 0.001 - 1000 |     0.001 - 1000 |  19 (24 possible) |
+------------+--------------+------------------+-------------------+
