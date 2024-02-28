.. folie documentation master file, created by
   sphinx-quickstart on Wed Dec  6 14:27:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to folie's documentation!
=================================





folie (Finding Optimal Langevin Inferred Equations) intends to provide a simple to use module for inference of Langevin equations from trajectories of collectives variables. Please refer to the documentation for a more complete description.


Installation
--------------

Installing the library is as simple as running

.. code-block:: bash

   pip install git+https://github.com/langevinmodel/folie.git


If you also want to install optionnal dependencies, run for the torch dependency


.. code-block:: bash

   pip install git+https://github.com/langevinmodel/folie.git[deep-learning]

or

.. code-block:: bash

   pip install git+https://github.com/langevinmodel/folie.git[finite-element]

to install dependencies for the finite element part of the library.


Getting Started
------------------

The general pattern of using folie is defining a model, loading trajectories data, fit the model and analyse the resulting model.





Table of contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   self
   howto
   for_developper

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   auto_examples/index

   tutorials.rst

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Performances

  statistical_performances.rst

.. toctree::
   :caption: API docs
   :maxdepth: 2

   api/dataloading
   api/models
   api/functions
   api/estimation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
