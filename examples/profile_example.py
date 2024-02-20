#!python3
# -*- coding: utf-8 -*-

"""
================================
Overdamped Langevin Estimation
================================

How to run a simple estimation
"""

import numpy as np
import folie as fl
import cProfile


prof = cProfile.Profile()

# Trouver comment on rentre les données
trj = np.loadtxt("example_2d.trj")
data = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
for i in range(1, trj.shape[1]):
    data.append(trj[:, i : i + 1])

fun = fl.functions.BSplinesFunction(knots=5)
model = fl.models.OverdampedFunctions(fun)
prof.enable()
estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))
model = estimator.fit_fetch(data)
prof.disable()
prof.dump_stats("likelihood.prof")
