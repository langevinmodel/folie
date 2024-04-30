import numpy as np
import matplotlib.pyplot as plt
import folie as fl
import csv 

# Define model
coeff=0.1*np.array([0,0,-4.5,0,0.1]) # coefficients of the free energy
free_energy = np.polynomial.Polynomial(coeff)
force_coeff=np.array([-coeff[1],-2*coeff[2],-3*coeff[3],-4*coeff[4]]) #coefficients of the force
force_function = fl.functions.Polynomial(deg=3,coefficients=force_coeff)
diff_function= fl.functions.Polynomial(deg=0,coefficients=np.asarray([0.5]))

# Define model to simulate and type of simulator to use
dt=1e-3
model_simu = fl.models.overdamped.Overdamped(force_function,diffusion=diff_function)
simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), dt) #, k=0.0, xstop=6.0)
ntraj=30
q0= np.empty(ntraj)
for i in range(len(q0)):
    q0[i]=0
# Calculate Trajectory
time_steps=10000
data = simulator.run(time_steps, q0, 1)
# Plot the trajecories
fig, axs = plt.subplots(1,1)
for n, trj in enumerate(data):
    axs.plot(trj["x"])
    axs.set_title("Trajectory")
    axs.set_xlabel("$timestep$")


# Parameters of the training
trainforce =fl.functions.Polynomial(deg=3,coefficients=np.asarray([1,1,1,1]))
traindiff = fl.functions.Polynomial(deg=0,coefficients=np.asarray([0.0]))
# trainmodel=fl.models.Overdamped(force = trainforce,diffusion=traindiff, has_bias=False)
domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min , data.stats.max , 10).ravel())
trainmodel= fl.models.OverdampedSplines1D(domain=domain)


fig, axs = plt.subplots(1, 2)
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()

axs[1].set_title("Diffusion")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
axs[1].grid()


xfa = np.linspace(-7.0, 7.0, 75)

axs[0].plot(xfa, model_simu.force(xfa.reshape(-1, 1)), label="Exact")
axs[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")
for name, transitioncls in zip(
    ["Euler", "Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
    [
        fl.EulerDensity,
        fl.OzakiDensity,
        fl.ShojiOzakiDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    estimator = fl.LikelihoodEstimator(transitioncls(trainmodel))
    res = estimator.fit_fetch(data)
    print(res.coefficients)
    # res.remove_bias()
    axs[0].plot(xfa, res.force(xfa.reshape(-1, 1)), label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), label=name)
axs[0].legend()
axs[1].legend()
plt.show()
