import numpy as np
import matplotlib.pyplot as plt
import folie as fl

coeff=0.1*np.array([0,0,-4.5,0,0.1])
free_energy = np.polynomial.Polynomial(coeff)
# diff_coeff = np.polynomial.Polynomial(np.array([]))
force_coeff=np.array([-coeff[1],-2*coeff[2],-3*coeff[3],-4*coeff[4]])
force_function = fl.functions.Polynomial(deg=3,coefficients=force_coeff)
diff_function= fl.functions.Polynomial(deg=2,coefficients=np.asarray([0.0,0.0,0.05])) # need a nonconstant diffusion function to see differences in Eul/Milst

# Plot of Free Energy and Force
x_values = np.linspace(-7, 7, 100)
fig, axs = plt.subplots(1, 2)
axs[0].plot(x_values,free_energy(x_values))
axs[1].plot(x_values,force_function(x_values.reshape(len(x_values),1)))
axs[0].set_title("Potential")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$V(x)$")
axs[0].grid()
axs[1].set_title("Force") 
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$F(x)$") 
axs[1].grid()

# Define model to simulate and type of simulator to use
dt=1e-3
model_simu = fl.models.overdamped.Overdamped(force_function,diffusion=diff_function)
simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), dt) #, k=0.0, xstop=6.0)

# initialize positions 
ntraj=4
q0= np.empty(ntraj)
for i in range(len(q0)):
    q0[i]=0.000001
# Calculate Trajectory
time_steps=50000
data = simulator.run(time_steps, q0, 1)


# Plot the resulting trajectories
# sphinx_gallery_thumbnail_number = 1
fig, axs = plt.subplots()
Eul_trajectory = []
for n, trj in enumerate(data):
    axs.plot(trj["x"])
    axs.set_title("Euler")
    axs.set_xlabel("$timestep$")
    axs.set_ylabel("$x(t)$")
    axs.grid()
    Eul_trajectory.append(trj["x"])

#######
# same with fixed seed to see if it works
######
data1 = simulator.run(time_steps, q0, 1)


# Plot the resulting trajectories
# sphinx_gallery_thumbnail_number = 1
fig, axs = plt.subplots()
Eul_trajectory1 = []
for n, trj1 in enumerate(data1):
    axs.plot(trj1["x"])
    axs.set_title("Euler1")
    axs.set_xlabel("$timestep$")
    axs.set_ylabel("$x(t)$")
    axs.grid()
    Eul_trajectory1.append(trj1["x"])

# ### =====================
# #   Same with Milstein integrator
# ## ======================
# # Calculate Trajectory
Mil_simulator = fl.simulations.Simulator(fl.simulations.MilsteinStepper(model_simu), dt)
Mil_data = Mil_simulator.run(time_steps, q0, 1)

print(data)
# Plot the resulting trajectories
# sphinx_gallery_thumbnail_number = 1
fig, axs = plt.subplots()
Mil_trajectory = []
for n, miltrj in enumerate(Mil_data):
    axs.plot(miltrj["x"])
    axs.set_title("Milstein")
    axs.set_xlabel("$timestep$")
    axs.set_ylabel("$x(t)$")
    axs.grid()
    Mil_trajectory.append(miltrj["x"])

fig, ax2 = plt.subplots()
for i in range(ntraj):
    # ax2.plot(abs(np.array(Eul_trajectory1[i])-np.array(Eul_trajectory[i])),c='r')
    ax2.plot(abs(np.array(Eul_trajectory1[i])-np.array(Mil_trajectory[i])))
ax2.set_title("diff in Eul-Mil traj with dt="+str(dt))




# ax2.plot(abs(np.array(Eul_trajectory1[1])-np.array(Eul_trajectory[1])),c='b')
# ax2.set_title("diff in Eul-Eul traj with dt="+str(dt))
# ax2.plot(abs(np.array(Eul_trajectory1[0])-np.array(Mil_trajectory[0])))



# fig, ax2 =plt.subplots(1,2)

print(max(abs(np.array(Eul_trajectory1[0])-np.array(Mil_trajectory[0]))))
print(max(abs(np.array(Eul_trajectory1[1])-np.array(Mil_trajectory[1]))))




# fig, axs = plt.subplots(1, 2)
# axs[0].set_title("Force")
# axs[0].set_xlabel("$x$")
# axs[0].set_ylabel("$F(x)$")
# axs[0].grid()

# axs[1].set_title("Diffusion Coefficeint")
# axs[1].set_xlabel("$x$")
# axs[1].set_ylabel("$D(x)$") 
# axs[1].grid()

# xfa = np.linspace(-7.0, 7.0, 75)
# model_simu.remove_bias()
# axs[0].plot(xfa, model_simu.force(xfa.reshape(-1, 1)), label="Exact")
# axs[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")
# for name, transitioncls in zip(
#    ["Euler", "Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
#    [
#        fl.EulerDensity,
#        fl.OzakiDensity,
#        fl.ShojiOzakiDensity,
#        fl.ElerianDensity,
#        fl.KesslerDensity,
#        fl.DrozdovDensity,
#    ],
# ):
#    estimator = fl.LikelihoodEstimator(transitioncls(fl.models.Overdamped(force_function,has_bias=True)))
#    res = estimator.fit_fetch(data)
#    print(res.coefficients)
#    res.remove_bias()
#    axs[0].plot(xfa, res.force(xfa.reshape(-1, 1)), label=name)
#    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), label=name)
# axs[0].legend()
# axs[1].legend()
plt.show() 