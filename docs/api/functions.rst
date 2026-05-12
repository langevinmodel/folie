==================
Spatial dependence
==================

.. currentmodule:: folie.functions


Parameteric Functions
=====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Constant
   ConstantForce
   Linear
   Polynomial
   Fourier
   Quadratic
   Quartic
   Quartic2D
   MultiWell1D
   Cosine
   ThreeWell
   BSplinesFunction
   sklearnBSplines
   FiniteElement
   Optimized1DLinearElement
   RadialBasisFunction
   MullerBrown
   RuggedMullerBrown
   LogExpPot
   ValleyRidgePotential
   SimpleValleyRidgePotential
   PotentialFunction
   ModelOverlay


Non-Parametric Functions
========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   sklearnWrapper
   KernelFunction


Base Classes
============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Function
   ParametricFunction
   FunctionSum


Analytical Potentials
=====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EntropicSwitch


Radial Basis Kernels
====================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   gaussian
   linear
   quadratic
   inverse_quadratic
   multiquadric
   inverse_multiquadric
   spline
   poisson_one
   poisson_two
   matern32
   matern52
   sigmoid
