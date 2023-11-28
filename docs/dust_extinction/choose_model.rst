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
(:class:`~dust_extinction.averages.RL85_MWGC`,
:class:`~dust_extinction.averages.RRP89_MWGC`,
:class:`~dust_extinction.averages.B92_MWAvg`,
:class:`~dust_extinction.averages.I05_MWAvg`,
:class:`~dust_extinction.averages.CT06_MWLoc`,
:class:`~dust_extinction.averages.CT06_MWGC`,
:class:`~dust_extinction.averages.GCC09_MWAvg`,
:class:`~dust_extinction.averages.F11_MWGC`,
:class:`~dust_extinction.averages.G21_MWAvg`;
:class:`~dust_extinction.averages.D22_MWAvg`;
Note the different valid wavelength ranges), the LMC
(:class:`~dust_extinction.averages.G03_LMCAvg`,
:class:`~dust_extinction.averages.G03_LMC2`) and the SMC
(:class:`~dust_extinction.averages.G03_SMCBar`).

One often used alternative to these straight average models is to use one of
the parameter dependent models with the average R(V) value.  For the Milky
Way, the usual average used is R(V) = 3.1.  See the next section.

+--------------+-------------+------------------+--------------+
| Model        | x range     | wavelength range |       galaxy |
|              | [1/micron]  | [micron]         |              |
+==============+=============+==================+==============+
| B92_MWAvg    | 1.3 - 2.9   |     0.34 - 0.78  |           MW |
+--------------+-------------+------------------+--------------+
| I05_MWAvg    |  0.13 - 0.8 |      1.24 - 7.76 |           MW |
+--------------+-------------+------------------+--------------+
| GCC09_MWAvg  | 0.3 - 10.96 |     0.0912 - 3.3 |           MW |
+--------------+-------------+------------------+--------------+
| G21_MWAvg    |  0.3125 - 1 |           1 - 32 |           MW |
+--------------+-------------+------------------+--------------+
| D22_MWAvg    |  0.2 - 1.25 |          0.8 - 4 |           MW |
+--------------+-------------+------------------+--------------+
| CT06_MWLoc   | 0.037 - 0.8 |      1.24 - 27.0 |   MW (Local) |
+--------------+-------------+------------------+--------------+
| RL85_MWGC    |  0.08 - 0.8 |      1.25 - 13.0 | MW (GCenter) |
+--------------+-------------+------------------+--------------+
| RRP89_MWGC   | 0.08 - 1.25 |       0.8 - 13.0 | MW (GCenter) |
+--------------+-------------+------------------+--------------+
| CT06_MWGC    | 0.037 - 0.8 |      1.24 - 27.0 | MW (GCenter) |
+--------------+-------------+------------------+--------------+
| F11_MWGC     |  0.05 - 0.8 |      1.28 - 19.1 | MW (GCenter) |
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

For MW type extinction, the
:class:`~dust_extinction.parameter_averages.G23` model should be considered as it
spectroscopically covers the far-ultraviolet (912 A) to mid-infrared (32 micron)
and is based on the spectroscopic extinction curves used for the
:class:`~dust_extinction.parameter_averages.GCC09`,
:class:`~dust_extinction.parameter_averages.F19`,
:class:`~dust_extinction.averages.G21_MWAvg`, and
:class:`~dust_extinction.parameter_averages.D22` studies.
For those who wish to bypass the python implementation of the 
:class:`~dust_extinction.parameter_averages.G23` model, tables for the range
of valid R(V) values with 0.1 steps are `available <https://stsci.box.com/v/ExtinctionTables>`_.

A more general model is :class:`~dust_extinction.parameter_averages.G16` as this
model encompasses the average measured behavior of extinction curves in the MW,
LMC, and SMC.  But it only covers wavelengths between 1150 A and 3 micron.
The :class:`~dust_extinction.parameter_averages.G16` model reduces
to the :class:`~dust_extinction.parameter_averages.F99` model with f\ :sub:`A`\ =
1.0.


+----------+-------------+--------------+------------------+--------------+
| Model    | Parameters  |  x range     | wavelength range |       galaxy |
|          |             |  [1/micron]  | [micron]         |              |
+==========+=============+==============+==================+==============+
| CCM89    |  R(V)       |   0.3 - 10.0 |        0.1 - 3.3 |           MW |
+----------+-------------+--------------+------------------+--------------+
| O94      |  R(V)       |   0.3 - 10.0 |        0.1 - 3.3 |           MW |
+----------+-------------+--------------+------------------+--------------+
| F99, F04 |  R(V)       |   0.3 - 10.0 |        0.1 - 3.3 |           MW |
+----------+-------------+--------------+------------------+--------------+
| VCG04    |  R(V)       |    3.3 - 8.0 |     0.125 - 0.31 |           MW |
+----------+-------------+--------------+------------------+--------------+
| GCC09    |  R(V)       |   3.3 - 11.0 |    0.0912 - 0.31 |           MW |
+----------+-------------+--------------+------------------+--------------+
| M14      |  R_5495     |   0.3 -  3.3 |       0.31 - 3.3 |      MW, LMC |
+----------+-------------+--------------+------------------+--------------+
| G16      | R(V)_A, f_A |   0.3 - 10.0 |        0.1 - 3.3 | MW, LMC, SMC |
+----------+-------------+--------------+------------------+--------------+
| F19      |  R(V)       |    0.3 - 8.7 |      0.115 - 3.3 |           MW |
+----------+-------------+--------------+------------------+--------------+
| D22      |  R(V)       |   0.2 - 1.25 |        0.8 - 5.0 |           MW |
+----------+-------------+--------------+------------------+--------------+
| G23      |  R(V)       | 0.032 - 11.0 |    0.0912 - 32.0 |           MW |
+----------+-------------+--------------+------------------+--------------+

Notes
-----

The :class:`~dust_extinction.parameter_averages.M14` models focus on refining
models in the optical, and use the
:class:`~dust_extinction.parameter_averages.CCM89` models for the NIR and the UV.
The :class:`~dust_extinction.parameter_averages.M14` models use
R_5495 = A(5485)/E(4405-5495), the spectroscopic equivalent to
band-integrated R(V); see the paper for discussion.  Because of a spurious
feature in the near UV caused by smoothly tying their optical to the
:class:`~dust_extinction.parameter_averages.CCM89` UV, only the NIR and
optical portions of the :class:`~dust_extinction.parameter_averages.M14`
models are provided here.

Grain Models
============

The models are based on dust grain models that are calculated based on
dust size, composition, and shape distributions.  The distributions
are constrained by observations of dust extinction, abundances, emission,
and polarization (usually a subset, not all).  One use of these models
is to provide extinction measurements at wavelengths not accessible
observationally (e.g., in the extreme UV below 912 A).

+--------------+----------------+------------------+--------------+
| Model        |    x range     | wavelength range |       galaxy |
|              |    [1/micron]  | [micron]         |              |
+==============+================+==================+==============+
| DBP90 MWRV31 | 0.00001 - 10.9 |  0.0918 - 100000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+
| WD01 MWRV31  |   0.0001 - 100 |     0.01 - 10000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+
| WD01 MWRV40  |   0.0001 - 100 |     0.01 - 10000 |  MW R(V)=4.0 |
+--------------+----------------+------------------+--------------+
| WD01 MWRV55  |   0.0001 - 100 |     0.01 - 10000 |  MW R(V)=5.5 |
+--------------+----------------+------------------+--------------+
| WD01 LMCAvg  |   0.0001 - 100 |     0.01 - 10000 |          LMC |
+--------------+----------------+------------------+--------------+
| WD01 LMC2    |   0.0001 - 100 |     0.01 - 10000 |  LMC2 Region |
+--------------+----------------+------------------+--------------+
| WD01 SMCBar  |   0.0001 - 100 |     0.01 - 10000 |          SMC |
+--------------+----------------+------------------+--------------+
| D03 MWRV31   | 0.0001 - 10000 |   0.0001 - 10000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+
| D03 MWRV40   | 0.0001 - 10000 |   0.0001 - 10000 |  MW R(V)=4.0 |
+--------------+----------------+------------------+--------------+
| D03 MWRV55   | 0.0001 - 10000 |   0.0001 - 10000 |  MW R(V)=5.5 |
+--------------+----------------+------------------+--------------+
| ZDA04 MWRV31 |  0.0001 - 1000 |    0.001 - 10000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+
|   C11 MWRV31 |   0.00001 - 25 |    0.04 - 100000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+
|   J13 MWRV31 |   0.00001 - 25 |    0.04 - 100000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+
|  HD23 MWRV31 |  0.000033 - 10 |      0.1 - 30000 |  MW R(V)=3.1 |
+--------------+----------------+------------------+--------------+

Shape Models
============

The models that focus on describing the full extinction curve shape are usually
used to fit measured extinction curves.  These models allow features in the
extinction curve to be measured (e.g., 2175 A bump or 10 micron silicate
feature).  The :class:`~dust_extinction.shapes.P92` is the most
general as it covers the a very broad wavelength range.  The
:class:`~dust_extinction.shapes.FM90` model covers the UV wavelength range
and has been extensively shown to fit all known UV extinction curves. 
The :class:`~dust_extinction.shapes.FM90_B3` model provides a variant
of the FM90 model that uses B3 instead of C3 as B3 = explicit 2175 A 
bump height = C3/gamma^2.
:class:`~dust_extinction.shapes.G21` model focuses on the NIR/MIR
wavelength range from 1-40 micron.

+------------+--------------+------------------+-------------------+
| Model      | x range      | wavelength range | # of parameters   |
|            | [1/micron]   | [micron]         |                   |
+============+==============+==================+===================+
| FM90       | 3.13 - 11.0  |    0.0912 - 0.32 |  6                |
+------------+--------------+------------------+-------------------+
| FM90_B3    | 3.13 - 11.0  |    0.0912 - 0.32 |  6                |
+------------+--------------+------------------+-------------------+
| P92        | 0.001 - 1000 |     0.001 - 1000 |  19 (24 possible) |
+------------+--------------+------------------+-------------------+
| G21        | 0.025 - 1    |           1 - 40 |  10               |
+------------+--------------+------------------+-------------------+
