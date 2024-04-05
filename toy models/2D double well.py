import numpy as np
import matplotlib.pyplot as plt
import folie as fl
from mpl_toolkits.mplot3d import Axes3D
import math
from math import *

x = np.linspace(-1,1,20)
y = np.linspace(-1,1,20)
input=np.transpose(np.array([x,y]))
diff_function= fl.functions.Polynomial(deg=0,coefficients=np.asarray([0.5]))
quartic= fl.functions.Quartic2D(a=6.0,b=4.0)
pot=quartic.potential(input)
ff=quartic.force(input) # returns x and y components of the force : x_comp =ff[:,0] , y_comp =ff[:,1]

# Plot Force function
U,V = np.meshgrid(ff[:,0],ff[:,1])
fig, ax =plt.subplots()
ax.quiver(x,y,U,V)
ax.set_title('Force')
plt.show()
print(quartic.domain)
fff=fl.functions.PotentialFunctionAdapter(quartic.force, dim=2)

model_simu=fl.models.overdamped.Overdamped(force=fff,diffusion=diff_function)
# model_simu=fl.models.overdamped.Overdamped(force=quartic.force,diffusion=diff_function)
simulator=fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), 1e-3)