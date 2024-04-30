import numpy as np
import matplotlib.pyplot as plt
import folie as fl
import csv 
from copy import deepcopy
import time 

# Define model parameters :  force and diffusion functions 
coeff=0.1*np.array([0,0,-4.5,0,0.1]) # coefficients of the free energy same as before but translated by 3 
free_energy = np.polynomial.Polynomial(coeff)
force_coeff=np.array([-coeff[1],-2*coeff[2],-3*coeff[3],-4*coeff[4]]) #coefficients of the free energy
force_function = fl.functions.Polynomial(deg=3,coefficients=force_coeff)
diff_function= fl.functions.Polynomial(deg=0,coefficients=np.asarray([0.5]))
# Plot of Free Energy and Force
x_values = np.linspace(-4, 10, 100)
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
simulator = fl.simulations.ABMD_Simulator(fl.simulations.EulerStepper(model_simu), dt, k=8.0, xstop=7.0)
ntraj=20
q0= np.empty(ntraj)
for i in range(len(q0)):
    q0[i]=0.1
# Calculate Trajectory
time_steps=35000
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
KMmodel=fl.models.Overdamped(force = trainforce,diffusion=traindiff, has_bias=True)
# domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min , data.stats.max , 10).ravel())
# trainmodel= fl.models.Overdamped(domain=domain)


fig, axs = plt.subplots(1, 2)
axs[0].set_title("Force Debaised")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()
axs[1].set_title("Diffusion")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
axs[1].grid()
xfa = np.linspace(-3.0, 9.0, 75)

time1 = time.time()
res_KM = fl.KramersMoyalEstimator(KMmodel).fit_fetch(data)
print('KM coeff'+str(KMmodel.coefficients))
model_simu.remove_bias()
axs[0].plot(xfa, model_simu.force(xfa.reshape(-1, 1)), label="Exact")
axs[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")
res_KM.remove_bias()
axs[0].plot(xfa, res_KM.force(xfa.reshape(-1, 1)), marker='x', label="KM")
axs[1].plot(xfa, res_KM.diffusion(xfa.reshape(-1, 1)), label="KM")


models = []
res_vec= []
# res_vec.append(res_KM)
markers = ['1', '2', '3', '4', '+', 'x']
names =["Euler", "Ozaki","ShojiOzaki", "Elerian", "Kessler", "Drozdov"]
for name, mark, transitioncls in zip(
    names,
    markers[:len(names)],
    [
        fl.EulerDensity,
        fl.OzakiDensity,
        fl.ShojiOzakiDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    trainforce =fl.functions.Polynomial(deg=3,coefficients=np.asarray([1,1,1,1]))
    traindiff = fl.functions.Polynomial(deg=0,coefficients=np.asarray([0.0]))
    trainmodel=fl.models.Overdamped(force = trainforce,diffusion=traindiff, has_bias=True)
    models.append(trainmodel)
    estimator = fl.LikelihoodEstimator(transitioncls(models[-1]))
    res = estimator.fit_fetch(data,coefficients0= KMmodel.coefficients)
    print(name, res.coefficients)
    res.remove_bias()
    res_vec.append(res)
    axs[0].plot(xfa, res.force(xfa.reshape(-1, 1)),marker= mark, label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), marker= mark,label=name)

axs[0].legend()
axs[1].legend()

for i in range(len(names)-1):
    flag_force= (res_vec[i].force(xfa.reshape(-1, 1)) == res_vec[i+1].force(xfa.reshape(-1, 1))).all()
    flag_diff= (res_vec[i].diffusion(xfa.reshape(-1, 1)) == res_vec[i+1].diffusion(xfa.reshape(-1, 1))).all()
    print(flag_force, flag_diff)  # apparently they are 
###########################################################################

# fig, axss = plt.subplots(1, 2)
# axss[0].set_title("Force Debiased")
# axss[0].set_xlabel("$x$")
# axss[0].set_ylabel("$F(x)$")
# axss[0].grid()
# axss[1].set_title("Diffusion")
# axss[1].set_xlabel("$x$")
# axss[1].set_ylabel("$D(x)$")
# axss[1].grid()
# xfa = np.linspace(-7.0, 7.0, 75)

# model_simu.remove_bias()
# axss[0].plot(xfa, model_simu.force(xfa.reshape(-1, 1)), label="Exact")
# axss[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")

# for name, transitioncls in zip(
#     ["Euler", "Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
#     [
#         fl.EulerDensity,
#         fl.OzakiDensity,
#         fl.ShojiOzakiDensity,
#         fl.ElerianDensity,
#         fl.KesslerDensity,
#         fl.DrozdovDensity,
#     ],
# ):
#     newestimator = fl.LikelihoodEstimator(transitioncls(trainmodel))
#     newres = newestimator.fit_fetch(data)
#     print(res.coefficients)
#     newres.remove_bias()
#     axss[0].plot(xfa, res.force(xfa.reshape(-1, 1)), label=name)
#     axss[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), label=name)
# axss[0].legend()
# axss[1].legend()
print('time needed='+str(time.time()-time1))
plt.show()
