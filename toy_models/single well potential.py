
import numpy as np
import matplotlib.pyplot as plt
import folie as fl

parab_force=np.polynomial.Polynomial([0.0,-4])
lin_force =fl.functions.Polynomial(polynom=parab_force)

x_values = np.linspace(-8, 8, 100)
y= parab_force(x_values)
fig, ax = plt.subplots()
ax.plot(x_values,y)
#plt.plot(x_values,y)
#plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].plot(x_values,parab_force(x_values))
#axs[1].plot(x_values,-d_poly(x_values))
axs[0].set_title("Potential")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$V(x)$")
axs[0].grid()

axs[1].set_title("Force") # i think should be diffusion coefficient
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$F(x)$") 
axs[1].grid()


#double_well_overdamped = fl.models.overdamped.Overdamped(poly)
#simulator = fl.simulations.ABMD_Simulator(fl.simulations.EulerStepper(double_well_overdamped), 1e-3, k=10.0, xstop=6.0)

model_simu = fl.models.overdamped.Overdamped(lin_force)
simulator = fl.simulations.ABMD_Simulator(fl.simulations.EulerStepper(model_simu), 5e-4, k=0.0, xstop=6.0)
#model_simu = fl.models.BrownianMotion(mu=0)
#simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu),1e-3)

# initialize positions 
ntraj=4
q0= np.empty(ntraj)
for i in range(len(q0)):
    q0[i]=-0

data = simulator.run(150000, q0, ntraj)

xmax = np.concatenate(simulator.xmax_hist, axis=1).T

# Plot the resulting trajectories
# sphinx_gallery_thumbnail_number = 1
fig, axs = plt.subplots(1,2)
for n, trj in enumerate(data):
    axs[0].plot(trj["x"])
    axs[1].plot(xmax[:, n])