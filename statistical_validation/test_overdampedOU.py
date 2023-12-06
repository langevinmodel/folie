import numpy as np
import folie as fl
import matplotlib.pyplot as plt

model_simu = fl.models.OrnsteinUhlenbeck()
data_dts = []
list_dts = [1e-3, 5e-3, 1e-2, 5e-2]
for dt in list_dts:
    simulator = fl.Simulator(fl.ExactDensity(model_simu), 1e-3)
    data_dts.append(simulator.run(5000, np.zeros((25,)), 25))

fig, axs = plt.subplots(1, len(model_simu.coefficients))

for name, transitioncls in zip(
    ["Exact", "Euler", "Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
    [
        fl.ExactDensity,
        fl.EulerDensity,
        fl.OzakiDensity,
        fl.ShojiOzakiDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    model = fl.models.OrnsteinUhlenbeck()
    estimator = fl.LikelihoodEstimator(transitioncls(model))
    coeffs_vals = np.empty((len(data_dts), len(model.coefficients)))
    for n, data in enumerate(data_dts):
        res = estimator.fit_fetch(data)
        coeffs_vals[n, :] = res.coefficients
    for n in range(len(axs)):
        axs[n].plot(list_dts, np.abs(coeffs_vals[:, n] - model_simu.coefficients[n]), "-+", label=name)
for n in range(len(axs)):
    axs[n].legend()
    axs[n].set_yscale("log")
    axs[n].grid()
    axs[n].set_xlabel("$\\Delta t")
    axs[n].set_ylabel("$|c-c_{real}|$")
plt.show()
