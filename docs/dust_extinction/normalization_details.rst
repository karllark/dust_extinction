#####################
Normalization Details
#####################

Most extinction curves in this package are normalized to A(V). This reflects
that the most useful normalization is to an absolute measure of the dust column.
Extinction curves have often also been normalized to E(B-V), mainly as
historically B and V photometry was available for the stars used to measure
extinction and the color excess between these two bands provided a (relative)
measure of the dust column. With the availability of JHK photometry (especially
after the 2MASS survey), it was possible to measure R(V) = A(V)/E(B-V) and
covert the curves to to the absolute normalization of A(V).

A(V) and E(B-V) normalization Subtleties
========================================

With the advent of using dust extinction curves to account for the effects of
dust extinction on stellar spectroscopy, the details of the wavelengths assumed
for the V and B band for normalized dust extinction curves is been of interest.

The majority of dust extinction curves are measured using optical photometry and
the wavelengths assumed for specific bands influence the application of dust
extinction curves for spectroscopic observations.

Simple measurements reveal that different authors have assumed different
wavelengths for the V and B bands. Other details in the construction of the
extinction curves can also impact the effective wavelengths of the V and B bands
(e.g., optical extinction shape in absence of V or B band photomtery). The table
below gives the wavelengths of the V and B bands by determining where the
A(lambda)/A(V) extinction curves have values of 1 (V band) and 1 + 1/R(V) (B
band). The R(V) values are the average R(V) for the average models and R(V) = 3.1
for the parameters averages and the grain models.

To obtain the "correct" R(V) values using monochromatic wavelengths, the B and V
wavelengths in the tables below need to be used.

Only models that are defined at V and B wavelengths are given in these tables.

Average Models
--------------

+--------------+--------------------+-----------------------------+------+
| Model        | A(lambda)/A(V) = 1 | A(lambda)/A(V) = 1 + 1/R(V) | R(V) |
+--------------+--------------------+-----------------------------+------+
|              | [micron]           | [micron]                    |      |
+==============+====================+=============================+======+
| B92_MWAvg    |             0.5500 |                      0.4400 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| G03_SMCBar   |             0.5500 |                      0.4422 | 2.74 |
+--------------+--------------------+-----------------------------+------+
| G03_LMCAvg   |             0.5500 |                      0.4399 | 3.41 |
+--------------+--------------------+-----------------------------+------+
| G03_LMC2     |             0.5500 |                      0.4369 | 2.76 |
+--------------+--------------------+-----------------------------+------+
| GCC09_MWAvg  |             0.5528 |                      0.4238 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| G24_SMCAvg   |             0.5530 |                      0.4440 | 3.02 |
+--------------+--------------------+-----------------------------+------+
| G24_SMCBumps |             0.5500 |                      0.4400 | 2.55 |
+--------------+--------------------+-----------------------------+------+
| C25_M31Avg   |             0.5409 |                      0.4430 | 3.20 |
+--------------+--------------------+-----------------------------+------+
| G26_M33Avg   |             0.5358 |                      0.4272 | 4.66 |
+--------------+--------------------+-----------------------------+------+

Parameter Average Models
------------------------

The values given below were calculated assuming R(V) = 3.1, but will be the same
for any R(V). The exception is for F19 where the normalization is to A(55), so
so R(55) = 3.1 was assumed. The A(55) values give the monochromatic value at
0.5500 micron while the A(V) values give the value averaged over the V band.

+--------------+--------------------+-----------------------------+------+
| Model        | A(lambda)/A(V) = 1 | A(lambda)/A(V) = 1 + 1/R(V) | R(V) |
+--------------+--------------------+-----------------------------+------+
|              | [micron]           | [micron]                    |      |
+==============+====================+=============================+======+
| CCM89        |             0.5495 |                      0.4405 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| O94          |             0.5495 |                      0.4399 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| F99          |             0.5414 |                      0.4354 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| F04          |             0.5414 |                      0.4352 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| M14          |             0.5495 |                      0.4405 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| F19          |             0.5501 |                      0.4399 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| G23          |             0.5494 |                      0.4392 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| G16          |             0.5414 |                      0.4354 | 3.10 |
+--------------+--------------------+-----------------------------+------+

Grain Models
------------

+--------------+--------------------+-----------------------------+------+
| Model        | A(lambda)/A(V) = 1 | A(lambda)/A(V) = 1 + 1/R(V) | R(V) |
+--------------+--------------------+-----------------------------+------+
|              | [micron]           | [micron]                    |      |
+==============+====================+=============================+======+
| DBP90        |             0.5512 |                      0.4456 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| WD01         |             0.5495 |                      0.4422 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| D03          |             0.5495 |                      0.4393 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| ZDA04        |             0.5517 |                      0.4456 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| C11          |             0.5511 |                      0.4213 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| J13          |             0.5512 |                      0.4336 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| HD23         |             0.5497 |                      0.4406 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| Y24          |             0.5511 |                      0.4416 | 3.10 |
+--------------+--------------------+-----------------------------+------+
| P24          |             0.5505 |                      0.4223 | 3.10 |
+--------------+--------------------+-----------------------------+------+

G23 Renormalization
===================

The G23 R(V) dependent average model requires a small, but significant
renormalization due to the specific observations that were used.

This model was constructed from 4 different literature samples (GCC09, F19, G21,
& D22), with only F19 having optical spectroscopy in addition to V photometry.
To have consistent values for all 4 samples, the R(V) values were derived for
the F19 using the observed V and JHK photometry. After publication, it was found
(M. Fouseneau 2025, private comm) that the G23 curves for different R(V) values
intersected at A(lambda)/A(V) = 0.9854 at lambda = 0.5493 micron. All the other
R(V) dependent average models intersect at A(lambda)/A(V) = 1 for different R(V)
values. This is a result of using the observed V band photometry to measure R(V)
and the optical spectroscopy for the A(lambda)/A(V) measurements. This indicates
that the observed V band photometry for the F19 sample is offset from the
optical spectroscopy by 1.5%. As the optical spectroscopy is from the Hubble
Space Telescope with the STIS spectrograph and the V band photometry is from
ground-based measurements, the optical spectroscopy will be more accurate.

The flux calibration offset between the V photometry and spectroscopy matters as
the F19 work used stellar atmosphere models for the unreddened comparison stars.
Thus, this work is sensitive to the absolute flux calibration of the different
observations and this has changed over the years (e.g., Bohlin 2014, AJ, 147,
127). This is in contrast to the other three works that used observed unreddened
stars and, thus, are only sensitivity to the relative calibration of the
instruments used.

Thus, the G23 model has been renormalized by dividing by 0.9854. In hindsight,
G23 should have derived V band fluxes from the optical spectroscopy instead of
using observed V photometry.

Computation Details
===================

The wavelengths given in the above tables are computed using
`utils/determine_norm_wavelength.py`.