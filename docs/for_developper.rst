#######################################################
For developper
#######################################################


Run
    >>> git clone git@github.com:langevinmodel/folie.git
    >>> cd folie
    >>> pip install -e .

to install the package for local developpement.


Input of trajectories data
==============================



Building an estimator of Langevin dynamics
============================================


Likelihood estimation: using Transition density
--------------------------------------------------


.. inheritance-diagram:: folie.EulerDensity folie.OzakiDensity folie.ShojiOzakiDensity folie.ElerianDensity folie.KesslerDensity folie.DrozdovDensity
    :top-classes: folie.estimation.transitionDensity.TransitionDensity


Model of Langevin Dynamics
=============================


Basic requirements of a model
-----------------------------------

A model should have a set of defaults coefficients. This give a possible set of initial coefficients for estimation but also allow to easily test and use the model.
