# Sme import as analytical.py
from .base import Function
from .._numpy import np
from ..domains import Domain
from .analytical import PotentialFunction

class ExamplePotential(PotentialFunction):
     """
     An example of potential
     """

     def __init__(self, a=1.0, dim=2):

         self.a = a * np.ones((dim,))  # Initialize parameters
         self.dim = dim  # That should probabibly be 2
         super().__init__()

     @property
     def coefficients(self):
         return np.array([self.a])  # Get access to parameters

     @coefficients.setter
     def coefficients(self, val):
         self.a = val[0]  # Set parameters

     # This is a convenience function for user, implementation is optionnal
     def potential(self, x , y ): # requires an np.mesh as input 
         return 2*x**2 - x*y + 2*y**2
     def force(self, X): #This is -grad(potential).  That should return an array of size (X.shape[0], dim).
         return self.a * np.ones_like(X)