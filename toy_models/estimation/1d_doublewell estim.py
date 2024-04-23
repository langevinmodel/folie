import numpy as np
import matplotlib.pyplot as plt
import folie as fl
import csv 

d=[]
header = []
file =open('/home/dbersano/folie/toy_models/estimation/datasets/data.csv', 'r')
header = next(csv.reader(file)) 
for row in file:
    d.append(float(row))
file.close() 
print(type(d[0]))
x = np.linspace(0,float(max(d)),len(d))
# print(traj[])\\

traj = dict([('x', np.array(d)),('dt', 0.001)])

fig, axs = plt.subplots(1, 2)
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()

axs[1].set_title("Diffusion")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
axs[1].grid()

print(traj.keys())

xfa = np.linspace(0, 7.0, 75)
data= fl.Trajectories(dt=0.001)
data.append(traj)
print(data[0].keys())
for name, transitioncls in zip(
    ["Euler","Ozaki", "ShojiOzaki", "Elerian", "Kessler", "Drozdov"],
    [      
        fl.EulerDensity,
        fl.OzakiDensity,
        fl.ShojiOzakiDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    estimator = fl.LikelihoodEstimator(transitioncls(fl.models.overdamped.OrnsteinUhlenbeck(has_bias=False)))
    res = estimator.fit_fetch(data)

    print(res.coefficients)
    # res.remove_bias()
    axs[0].plot(xfa, res.force(xfa.reshape(-1, 1)), label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), label=name)
axs[0].legend()
axs[1].legend()
plt.show()
print(header)