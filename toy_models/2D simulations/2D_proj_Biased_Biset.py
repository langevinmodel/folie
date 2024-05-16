import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import sympy as sym
# from sympy import *

x = np.linspace(-1.8,1.8,36)
y = np.linspace(-1.8,1.8,36)
input=np.transpose(np.array([x,y]))

diff_function= fl.functions.Polynomial(deg=0,coefficients=np.asarray([0.5]) * np.eye(2,2))
a,b = 0.5, 1.0
quartic2d= fl.functions.Quartic2D(a=a,b=b)
exx = fl.functions.analytical.My_Quartic2D(a=a,b=b)

X,Y =np.meshgrid(x,y)

# Plot potential surface 
pot = exx.potential(X,Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,pot, rstride=1, cstride=1,cmap='jet', edgecolor = 'none')

# Plot Force function
ff=quartic2d.force(input) # returns x and y components of the force : x_comp =ff[:,0] , y_comp =ff[:,1]
U,V = np.meshgrid(ff[:,0],ff[:,1])
fig, ax =plt.subplots()
ax.quiver(x,y,U,V)
ax.set_title('Force')
# plt.show()
print(quartic2d.domain)
fff=fl.functions.Quartic2DForce(exx.force, dim=2)
##Definition of the Collective variable function of old coordinates 
def colvar (x,y):
    gradient = np.array([1,1])
    return x + y , gradient    #need to return both colvar function q=q(x,y) and gradient (dq/dx,dq/dy)

dt = 1e-3
model_simu=fl.models.overdamped.Overdamped(force=quartic2d,diffusion=diff_function)
simulator=fl.simulations.ABMD_2D_to_1DColvar_Simulator(fl.simulations.EulerStepper(model_simu), dt,colvar=colvar,k=25.0,qstop=1.2)

# initialize positions 
ntraj=50
q0= np.empty(shape=[ntraj,2])
for i in range(ntraj):
    # q0[i]=(0.8,-0.8)
    for j in range(2):
        q0[i][j]=-1.2

####################################
##       CALCULATE TRAJECTORY     ##
####################################

time_steps=10000
data = simulator.run(time_steps, q0,save_every=1)  
#xmax = np.concatenate(simulator.xmax_hist, axis=1).T

# Plot the resulting trajectories
fig, axs = plt.subplots()
for n, trj in enumerate(data):
    axs.plot(trj["x"][:,0],trj["x"][:,1])
    axs.spines['left'].set_position('center')
    axs.spines['right'].set_color('none')
    axs.spines['bottom'].set_position('center')
    axs.spines['top'].set_color('none')
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')
    axs.set_xlabel("$X(t)$")
    axs.set_ylabel("$Y(t)$")
    axs.set_title("X-Y Trajectory")
    axs.set_xlim(-1.8,1.8)
    axs.set_ylim(-1.8,1.8)
    # ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    axs.grid()

"""fig, axs = plt.subplots(1,2)
for n, trj in enumerate(data):
    axs[0].plot(trj["x"][:,0],trj["x"][:,1])
    #axs[1].plot(xmax[:, n])
    axs[1].set_xlabel("$timestep$")
    axs[1].set_ylabel("$x(t)$")
    axs[1].grid()"""
# plot x,y Trajectories in separate subplots
fig,bb =  plt.subplots(1,2)
for n, trj in enumerate(data):
    bb[0].plot(trj["x"][:,0])
    bb[1].plot(trj["x"][:,1])

# Set visible  axis
    bb[0].spines['right'].set_color('none')
    bb[0].spines['bottom'].set_position('center')
    bb[0].spines['top'].set_color('none')
    bb[0].xaxis.set_ticks_position('bottom')
    bb[0].yaxis.set_ticks_position('left')
    bb[0].set_xlabel("$timestep$")
    bb[0].set_ylabel("$X(t)$")

# Set visible axis
    bb[1].spines['right'].set_color('none')
    bb[1].spines['bottom'].set_position('center')
    bb[1].spines['top'].set_color('none')
    bb[1].xaxis.set_ticks_position('bottom')
    bb[1].yaxis.set_ticks_position('left')
    bb[1].set_xlabel("$timestep$")
    bb[1].set_ylabel("$Y(t)$")

    bb[0].set_title("X Dynamics")
    bb[1].set_title("Y Dynamics")
    # bb.grid()


#########################################
#  PROJECTION ALONG CHOSEN COORDINATE   #
#########################################

# Choose unit versor of direction 
u = np.array([1,1])
u_norm= (1/np.linalg.norm(u,2))*u
w = np.empty(shape=(len(trj["x"]),1))

proj_data = fl.data.trajectories.Trajectories(dt=dt) # create new Trajectory object in which to store the projected trajectory dictionaries

fig, axs =plt.subplots()
for n, trj in enumerate(data):
    for i in range(len(trj["x"])):
        w[i]=np.dot(trj["x"][i],u_norm)
    proj_data.append(w)
    axs.plot(proj_data[n]["x"])
    axs.set_xlabel("$timesteps$")
    axs.set_ylabel("$w(t)$")
    axs.set_title("trajectory projected along $u =$"  + str(u) + " direction")
    axs.grid()

#######################################
##          MODEL TRAINING           ##
#######################################

fig, axs = plt.subplots(1, 2)
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()
axs[1].set_title("Diffusion")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
axs[1].grid()
#### Possible models 

xfa = np.linspace(-1.3, 1.3, 75)
exact = (xfa** 2 - 1.0) ** 2
domain = fl.MeshedDomain.create_from_range(np.linspace(proj_data.stats.min , proj_data.stats.max , 40).ravel())
# domain = fl.MeshedDomain.create_from_range(np.linspace(min(xfa) , max(xfa) , 10).ravel())
trainmodel = fl.models.OverdampedSplines1D(domain=domain)

res_vec=[]
fig, axs = plt.subplots(1, 2)
axs[0].plot(xfa, exact, label="Exact")
axs[1].plot(xfa, exact, label="Exact")
model_simu.remove_bias()
for name, transitioncls in zip(
    ["Euler"],#, "Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
    [
        fl.EulerDensity,
        # fl.OzakiDensity,
        # fl.ShojiOzakiDensity,
        # fl.ElerianDensity,
        # fl.KesslerDensity,
        # fl.DrozdovDensity,
    ],
):
    estimator = fl.LikelihoodEstimator(transitioncls(deepcopy(trainmodel)))
    res = estimator.fit_fetch(proj_data)
    print(res.coefficients)
    res_vec.append(res)
    res.remove_bias()
    axs[0].plot(xfa, res.force(xfa.reshape(-1, 1)), label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), label=name)

axs[0].legend()
axs[1].legend()


for i in range(len(res_vec)-1):
    flag_force= (res_vec[i].force(xfa.reshape(-1, 1)) == res_vec[i+1].force(xfa.reshape(-1, 1))).all()
    flag_diff= (res_vec[i].diffusion(xfa.reshape(-1, 1)) == res_vec[i+1].diffusion(xfa.reshape(-1, 1))).all()
    print(flag_force, flag_diff)  # apparently they are 
plt.show()