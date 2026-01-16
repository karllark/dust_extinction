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

A(V) normalization Subtleties
=============================

With the advent of using dust extinction curves to account for the effects of
dust extinction on stellar spectroscopy, the details of the wavelengths assumed
for the V band normalized dust extinction curves has been of interest.

A simple measurement reveals that different authors have assumed different
wavelengths for the V band. Other details about in the construction of the
extinction curves can also impact the effective wavelength of the V band (e.g.,
optical extinction shape in absence of V band photomtery). The table below gives
the wavelengths where the normalized dust extinction curves are equal to one.
Only models that are defined in the V band are given in this table.

Average Models
--------------

+--------------+--------------------+
| Model        | A(lambda)/A(V) = 1 | 
|              | [micron]           |
+==============+====================+
| B92_MWAvg    | 0.5500             |
+--------------+--------------------+
| G03_SMCBar   | 0.5500             |
+--------------+--------------------+
| G03_LMCAvg   | 0.5500             |
+--------------+--------------------+
| G03_LMC2     | 0.5500             |
+--------------+--------------------+
| GCC09_MWAvg  | 0.5528             |
+--------------+--------------------+
| G24_SMCAvg   | 0.5530             |
+--------------+--------------------+
| G24_SMCBumps | 0.5500             |
+--------------+--------------------+
| C25_M31Avg   | 0.5409             |
+--------------+--------------------+
| G26_M33Avg   | 0.5358             |
+--------------+--------------------+

Parameter Average Models
------------------------

Specifically for R(V) = 3.1, except in the case of F19 which is based on optical
spectroscopy and normalized to F(55).
The R(55) values give the monochromatic
value at 0.5500 micron while the R(V) values give the value averaged over
the V band.

+--------------+--------------------+
| Model        | A(lambda)/A(V) = 1 | 
|              | [micron]           |
+==============+====================+
| CCM89        | 0.5495             |
+--------------+--------------------+
| O94          | 0.5495             |
+--------------+--------------------+
| F99          | 0.5414             |
+--------------+--------------------+
| F04          | 0.5414             |
+--------------+--------------------+
| M14          | 0.5495             |
+--------------+--------------------+
| F19          | 0.5500             |
+--------------+--------------------+
| G23          | 0.5493             |
+--------------+--------------------+

Grain Models
------------

For the default, generally the MW average.

+--------------+--------------------+
| Model        | A(lambda)/A(V) = 1 | 
|              | [micron]           |
+==============+====================+
| DBP90        | 0.5512             |
+--------------+--------------------+
| WD01         | 0.5495             |
+--------------+--------------------+
| D03          | 0.5495             |
+--------------+--------------------+
| ZDA04        | 0.5517             |
+--------------+--------------------+
| C11          | 0.5511             |
+--------------+--------------------+
| J13          | 0.5512             |
+--------------+--------------------+
| HD23         | 0.5497             |
+--------------+--------------------+
| Y24          | 0.5511             |
+--------------+--------------------+
| P24          | 0.5505             |
+--------------+--------------------+

G23 Renormalization
===================

The G23 R(V) dependent average model requires a small, but significant
renormalization due to the specific observations that were used.

This model was constructed from 4 different literature samples (GCC09, F19, G21,
& D22), with only F19 having optical spectroscopy in addition to V photometry.
To have consistent values for all 4 samples, the R(V) values were derived for
the F19 using the observed V and JHK photometry. After publication, it was found
(M. Fouseneau 2025, private comm.) that the G23 curves for different R(V) values
intersected at A(lambda)/A(V) = 0.9854 at lambda = 0.5493 micron. All the other
R(V) dependent average models intersect at A(lambda)/A(V) = 1 for different R(V)
values. This is a result of using the observed V band photometry to measure R(V)
and the optical spectroscopy for the A(lambda)/A(V) measurements. This indicates
that the observed V band photometry for the F19 sample is offset from the
optical spectroscopy by 1.5%. As the optical spectroscopy is from the Hubble
Space Telescope with the STIS spectrograph and the V band photometry is from
ground-based measurements, the optical spectroscopy will be more accurate. Thus,
the G23 model has been renormalized by dividing by 0.9854. In hindsight, G23
should have derived V band fluxes from the optical spectroscopy instead of using
observed V photometry.

