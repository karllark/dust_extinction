1.6.dev (unreleased)
================

- add G25 M33Avg model
- add C25 M31Avg model
- Refactor model input unit conversion and wavenumber bounds checking:

  - The models' evaluate methods are no longer meant to be called directly.

  - The BaseExtModel class now overrides the prepare_inputs method so that
    unit conversion and bounds checking is done prior to calling the subclass'
    evaluate method. Subclasses are no longer responsible for doing the unit
    conversion and bounds checking themselves.

  - Use Astropy's built-in input model units handling.

- fully move to pyproject.toml

  - setup.cfg removed
  - other misc cleanup

1.5 (2024-08-16)
================

- Add Y24 dust grain model
- Added JOSS paper
- Added development documentation
- Other documentation updates

1.4.1 (2024-05-21)
================

- Minor updates to G24 models

1.4 (2024-05-07)
================

- Added G24 Average and Bumps average models

1.3 (2023-11-28)
================

- Added HD23 dust grain model
- Added FM90_B3 shape model
- updated docs
- removed astropy 6.0 deprecated features

1.2 (2023-04-04)
================

- Added G23 parameter average model

1.1 (2022-03-30)
================

- Added D22_MWAvg average and D22 parameter average models

1.0 (2021-04-01)
================

- more NIR/MIR average models added (RRP89, G21_MWAvg)
- NIR/MIR shape model (G21)
- dust grain models added (WD01, D03, J13, ZDA04)

0.9 (2020-06-05)
================

- NIR/MIR average models added
  (RL85_MWGC, I05_MWAvg, CT06_MWLoc, CT06_MWGC, F11_MWGC)
- updates to be remove astropy 4.0 deprecated modeling features
- Documentation updates
- Many tests consolidated
- Bugfixes

0.8 (2019-08-20)
================

- VCG04, GCC09, FM19 models added

0.7 (2019-02-11)
================

- GCC09_MWAvg, O94, F04, M14 models added
- AxAvtoExv helper model added
- significant updates to the documentation
- extinction versus attenuation documentation added
- models reorganized into 3 categories (*API change*)
- made a new base class for R(V)_A, f_A dependent models (G16, expecting more)

0.6 (2017-11-15)
================

- minor updates to release version, notes

0.5 (2017-11-15)
================

- initial release
