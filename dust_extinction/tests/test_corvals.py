import pytest
import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import CCM89, O94, F99, F04, M14, F19

# fmt: off
models_corvals = {
    CCM89: {
        "Rv": 3.1,
        "x_values": [2.78, 2.27, 1.82, 1.43, 1.11, 0.80, 0.63, 0.46],
        "y_values": [1.569, 1.322, 1.000, 0.751, 0.479, 0.282, 0.190, 0.114],
        "atol": 1e-2,
        # values from Table 3 of Cardelli et al. (1989)
        #   ignoring the last value at L band as it is outside the
        #   valid range for the relationship
        #  updated for correction for incorrect value in the table for B band
        #    correction from Geoff Clayton via email
    },
    O94: {
        "Rv": [3.1, 3.1, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x_values": [[2.939, 2.863, 2.778, 2.642, 2.476, 2.385, 2.275, 2.224, 2.124, 2.000,
                      1.921, 1.849, 1.785, 1.718, 1.637, 1.563, 1.497, 1.408, 1.332, 1.270],
                     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46],
                     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46],
                     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46],
                     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46],
                     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46],
                     [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46]],
        # convert the table values of E(l-1.5)/E(2.2-1.5) to A(x)/A(V)
        "y_values": [[(np.array([1.725, 1.651, 1.559, 1.431, 1.292, 1.206, 1.100, 1.027, 0.907, 0.738,
                                 0.606, 0.491, 0.383, 0.301, 0.190, 0.098, -0.004, -0.128, -0.236, -0.327])
                       * (1.27992402 - 0.79848375)) + 0.79848375],
                     [5.23835484, 4.13406452, 3.33685933, 2.77962453, 2.52195399, 2.84252644,
                      3.18598916, 2.31531711, 0.28206957, 0.19200814, 0.11572348],
                     [9.407, 7.3065, 5.76223881, 4.60825807, 4.01559036, 4.43845534,
                      4.93952892, 3.39275574, 0.21678862, 0.14757062, 0.08894094],
                     [5.491, 4.32633333, 3.48385202, 2.8904508, 2.6124774, 2.9392494,
                      3.2922643, 2.38061642, 0.27811315, 0.18931496, 0.11410029],
                     [3.533, 2.83625, 2.34465863, 2.03154717, 1.91092092, 2.18964643,
                      2.46863199, 1.87454675, 0.30877542, 0.21018713, 0.12667997],
                     [2.3582, 1.9422, 1.66114259, 1.51620499, 1.48998704, 1.73988465,
                      1.97445261, 1.57090496, 0.32717278, 0.22271044, 0.13422778],
                     [1.575, 1.34616667, 1.20546523, 1.17264354, 1.20936444, 1.44004346,
                      1.64499968, 1.36847709, 0.33943769, 0.23105931, 0.13925965]],
        "atol": [6e-2, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        # 1st set of values from Bastiaansen (1992) Table 6 (testing versus that average)
    },
    F99: {
        "Rv": 3.1,
        "x_values": [0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846],
        "y_values": np.array([0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591]) / 3.1,
        "atol": 1e-3,
        # from Fitzpatrick (1999) Table 3
    },
    F04: {
        "Rv": 3.1,
        "x_values": [0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846],
        "y_values": np.array([0.185, 0.772, 2.688, 3.055, 3.805, 4.315, 6.456, 6.781]) / 3.1,
        "atol": 1e-3,
        # from Fitzpatrick (1999) Table 3
        # keep optical from Fitzpatrick (1999),
        # replce NIR with Fitzpatrick (2004) function for Rv=3.1:
        # (0.63*3.1 - 0.84)*x**1.84
    },
    M14: {
        # using R5495 = 3.1
        "Rv": 3.1,
        "x_values": [0.5, 2.0],
        "y_values": [0.1323, 1.141],
        "atol": 1e-3,
    },
    F19: {
        "Rv": 3.1,
        "x_values": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        # convert from E(x-V)/E(B-V) to A(x)/A(V)
        "y_values": (np.array([-1.757, -0.629, 0.438, 2.090, 4.139, 5.704, 4.904, 5.684, 7.150]) + 3.1) / 3.1,
        "atol": 1e-3,
        # use x values from Fitzpatrick et al. (2000) Table 3
    },
}
# fmt: on


@pytest.mark.parametrize(
    ("model_class", "test_parameters"),
    sorted(models_corvals.items(), key=lambda x: str(x[0])),
)
def test_corvals(model_class, test_parameters):
    Rv_vals = np.atleast_1d(test_parameters["Rv"])
    for k, Rv in enumerate(Rv_vals):
        tol = np.atleast_1d(test_parameters["atol"])[k]
        if len(Rv_vals) > 1:
            x_vals = test_parameters["x_values"][k]
            y_vals = test_parameters["y_values"][k]
        else:
            x_vals = test_parameters["x_values"]
            y_vals = test_parameters["y_values"]
        x_vals = np.array(x_vals).flatten() / u.micron
        y_vals = np.array(y_vals).flatten()

        # instantiate extinction model
        tmodel = model_class(Rv=Rv)

        # test array evaluation
        np.testing.assert_allclose(tmodel(x_vals), y_vals, atol=tol)

        # test single value evalutation
        for x, y in zip(x_vals, y_vals):
            np.testing.assert_allclose(tmodel(x), y, atol=tol)
            np.testing.assert_allclose(tmodel.evaluate(x, Rv), y, atol=tol)
