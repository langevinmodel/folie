FOLIE - JOSS
===

---
title: 'folie: Finding Optimal Langevin Inferred Equations'
tags:
  - Python
  - molecular dynamics
  - Langevin dynamics
  - effective dynamics
authors:
  - name: Hadrien Vroylandt
    orcid: 0000-0002-2443-5901
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 2
  - name: Daniele Bersano
    affiliation: 3
  - name: Jérôme Hénin
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3


affiliations:
 - name: Université Caen Normandie, ENSICAEN, CNRS, Normandie Univ, GREYC UMR 6072, 14000 Caen, France
   index: 1
   ror: 043749971
 - name: Sorbonne Université, Institut des sciences du calcul et des donn ́ées, ISCD - F-75005 Paris, France
   index: 2
 - name: Université Paris Cité, CNRS, Laboratoire de Biochimie Théorique UPR 9080, 75005, Paris, France
   index: 3
   ror: 00nvjgv40
   ror: 05abgg682
date: 2 April 2025
bibliography: paper.bib

---

# Summary


This paper introduces FOLIE (Finding Optimal Langevin Inferred Equations), a versatile Python library designed to facilitate the analysis of high-dimensional molecular simulation trajectories in terms of low-dimensional dynamics. FOLIE enables scientists to fit low-dimensional stochastic differential equations (SDEs) to projected high-dimensional data, thereby extracting maximum dynamical insight from the simulations. Key features of FOLIE include robust estimation techniques, comprehensive analysis tools, and simulation capabilities to create synthetic datasets. Its highly modular architecture allows for the implementation of diverse SDEs, time discretization methods, and energy landscapes. By leveraging FOLIE, researchers can effectively extrapolate low-dimensional kinetics from limited simulation data, enhancing their ability to understand and predict complex molecular dynamics.


# Statement of need

Quantitative predictions of rare events in molecular and materials simulations suffer from the curse of dimensionality and the prohibitive cost of simulating relevant time scales, starting from microscopic time steps (on the order of $10^{-15} s$).
Such rare events include chemical reactions in chemistry, drug unbinding in pharmacology, and phase transitions in materials.
A strategy to make such predictions tractable is to construct intermediate, low-dimensional kinetic models of the time evolution of the system.
Such low-dimensional models are constructed based on a projection of the full dynamics onto a reduced set of collective variables. This set must fulfill two constraints: describe the process of interest, and be informative enough to capture the long-time dynamics of this process.

FOLIE is designed to allow easy and efficient inference of such models from projected molecular simulations. There exits several software packages performing related tasks, but FOLIE differ mainly by its flexibility in the description of the energy landscape and its modular construction for the estimation task.
[DeepTime](https://deeptime-ml.github.io) (previously pyEmma) [@hoffmann2021deeptime] is an equivalent for discrete Markovian processes. [pymle](https://github.com/jkirkby3/pymle) fits continuous SDEs but imposes restraints on the underlying energy landscapes, which are better suited to econometrics and financial markets [@kirkby2024pymle].
[OptLE](https://github.com/physix-repo/optle) [@PalacioRodriguez2022] is an experimental Fortran package that inspired this work. It focuses on method development and was not designed for flexibility or scalability.
[pyOptLE](https://github.com/jhenin/pyOptLE) was a first attempt at improving scalability, with a very limited scope. [StochasticForceInference](https://github.com/ronceray/StochasticForceInference) and [UnderdampedLangevinInference](https://github.com/ronceray/UnderdampedLangevinInference) are codes from the same group focusing on the similar task of fitting continuous SDE but focusing on biological applications and such having less flexibility on the underlying energy landscapes.


# Theoretical background


## Langevin Models

Let's consider the dynamics of a low-dimensional collective variable $q$. There is a set of possible Langevin models to describe this dynamics[@PalacioRodriguez2022,@giradier2023]. Projecting the high-dimensionnal dynamics onto the collective variable leads to the generalized Langevin equation[@vroylandt2022a], that writes for a single variable
\begin{equation*}
    \ddot{q}= - \frac{1}{m(q)} \frac{\partial A(q)}{\partial q}+ k_BT\frac{\partial m(q)^{-1}}{\partial q}- \int_0^t  \Gamma(s)\dot{q}(t-s) \,\mathrm{d} s + R(t)
\end{equation*}
Here, $m(q)$ represent a position-dependent effective mass,  $-\frac{\partial A(q)}{\partial q}$ is the conservative force field in which the dynamics takes place, and the effective free energy surface is $A(q) = -k_BT \log (\rho_{eq}(q))$ where $\rho_{eq}$ is the invariant distribution of the dynamics. $\Gamma(s)$ is a memory kernel, a time dependent function describing the correlation of the velocity at time $t$ with itself at a previous time  $t-s$ and $R(t)$ is a random force. The random force is usually assumed to be related to the memory kernel according to the fluctuation-dissipation theorem $\langle R(0)R(t)\rangle = \frac{k_BT}{m} \Gamma(t)$, with $m=\int m(q)\rho_{eq}(q) \mathrm{d}q$, even if this relation is approximative in this framework[@vroylandt2022a].


Assuming that the timescale of evolution of the collective variable is slow with respect to its environnement; we can take the assumption of a Dirac kernel $\Gamma(s)=\gamma \delta(s)$, the fluctuation dissipation theorem being now valid. We then obtain the memory-less (Markovian) Standard Langevin equation 
\begin{equation}
\ddot{q}= - \frac{1}{m(q)} \frac{\partial A(q)}{\partial q}+ k_BT\frac{\partial m(q)^{-1}}{\partial q} -\frac{\gamma}{m(q)} \dot{q}+ \sqrt{\frac{2k_BT \gamma}{m(q)}}\eta(t)
\end{equation}
where $\eta(t)$ is now a standard Gaussian noise. One commomn assumption that we do not take here is a position independent effective lass.
If one were to consider the dynamics for underdamped motion on a time scale $\tau \gg\frac{m}{\gamma}$ non equilibrium fluctuations are quickly damped. This entitles us to improperly consider $\ddot{q} \approx 0$ leading to the Overdamped Langevin Equation 
\begin{equation}
\dot{q}= -\beta D(q)\frac{\partial A(q)}{\partial q}+ \frac{\partial D(q)}{\partial q} + \sqrt{2D(q)}\eta(t)
\end{equation}
with the definition of the diffusion profile from $D(q) = \frac{k_BT}{m(q)\gamma}$ and using $\beta = \frac{1}{k_BT}$.
This last model sensibly simplify the mathematical structure being a first order differential equation. The current state of the library is to infer reduced Langevin models starting from projected simulation trajectories, focusing so far on the overdamped case.

## Kinetic model optimization by a maximum-likelihood approach

In order to construct the optimal Langevin model to describe the dynamics in this lower-dimensional projection, we start by defining $\mathcal{L}( \vec{q}|\theta)$, the likelihood of observing the trajectory data $\vec{q}$ if they were to be generated by one of the specified Langevin models parameterized by $\theta$. Here $\theta$ stands for a compact way of regrouping both the drift $F(q)=-\beta D(q)\frac{\partial A(q)}{\partial q}+ \frac{\partial D(q)}{\partial q}$ and position dependent diffusion $D(q)$. 
Consequently the best values of $\theta$ is found by maximising the likelihood of the observed trajectory
\begin{equation}
    \hat{\theta}= \underset{\theta}{\mathrm{argmax}} \; \mathcal{L}(\theta).
\end{equation}
 
The analytical shape of the likelihood is not always known a priory but, restricting to the overdamped case, it will be the product of short time transition probability between consecutive trajectory points due to the markovian property of the equation itself [@PalacioRodriguez2022]. Using  the transition density $p_\theta(q_{i+1},t_{i+1}|q_{i},t_{i})$, i.e. the probability of being in position $q_i$ at time $t_i$ starting from $q_{i-1}$ at time $t_{i-1}$ , for given initial conditions $(q_0,t_0)$, the likelihood can be written as
\begin{equation}
\mathcal{L}(\vec{q}|\theta ) = \prod_{i=0}^{N-1} p_\theta(q_{i+1},t_{i+1}|q_{i},t_{i})
\end{equation}
from which follows that the log-likelihood of the trajectory is :
\begin{equation} \label{eq:eq:log_likelihood_general}
    \log{\mathcal{L}}(\vec{q}|\theta) = \sum_{i=0}^{N-1} \log \left[ p_\theta(q_{i+1},t_{i+1}|q_{i},t_{i}) \right]
\end{equation}

The precise form of the transition density is contingent on the choice of a specific time discretization (essentially related to the choice of an integrator) of the continuous SDE, for which several possiblities are implemented within FOLIE as explained below.

# Implementation

## Features

The library has been implemented with a modular structure, so that users can assemble the relevant components in simple Python scripts tailored to their needs.

- **Model of Overdamped Langevin Dynamics**
    Several models of possible implementation of Overdamped Langevin equations are present starting from the parent python class $\texttt{Overdamped}$, followed by the implementation of particular cases such as $\texttt{BrownianMotion}$ and $\texttt{OrnsteinUhlenbeck}$. The underdamped case is still under development.
    
- **Model for the force and diffusion coefficient functions**
    In addition to the dynamics, building the transition density and likelihood estimator requires the specification of the drift and space-dependent diffusion coefficient. To that effect, the $\texttt{Function}$ class has been implemented, which offers functional forms such as polynomials, splines, etc.

- **Transition densities** 
    Different methods aimed at approximating the form of the propagator are implemented[@iacus2008a]. They require as input a $\texttt{Model}$ object whose drift and diffusion will be used to compute the mean, variance, and, in case of the Elerian transition probability density, the additional parameters to obtain the relative likelihood.
    These probability densities are later fed to the $\texttt{Likelihood}$ estimator object, which optimizes the drift and diffusion parameters.
    
 - **Estimation**
    Given as input the probability densities used to compute the likelihood of the observed trajectories, an estimator object is created.
    The class of estimator objects playing a central role in performing the addressed task is the $\texttt{LikelihoodEstimator}$ class, within it, the MLE estimators are recovered by making use of the [optimize.minimize()](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.minimize.html) method from the scipy library applied to the negative of the log-likelihood function eq.(\ref{eq:log_likelihood_general}).
    

- **Simulation**
    In addition to the principal purpose of training the maximum likelihood estimator for a given set of input trajectories, the FOLIE module also allows to simulate trajectories, which is useful for creating synthetic data to be used when developing methods. This is achieved by first specifying the model of the Langevin equation guiding the evolution of the system through a suitable $\texttt{Overdamped}$ object. Then this passed to the $\texttt{Simulator}$ (or possibly $\texttt{BiasedSimulator}$) class specifying the integration timestep employed. Finally, the dataset is generated by calling $\texttt{Simulator.run()}$.



## Initial guess for parameters

Optimization of the likelihood requires an initial guess for the drift and diffusion parameter. In FOLIE, we use a Kramers-Moyal estimation to provide such a guess. The Kramers-Moyal estimator consists of the empirical estimation of the first two coefficients of the Kramers-Moyal expansion for the evolution (Master) equation associated with the equilibrium probability density $\rho_{eq}(q)$, $\textit{i.e}$ the Fokker-Plank equation [@Risken1996].

From a a set of trajectories, the Kramers-Moyal estimator computes the drift term (respectively the diffusion term) as the first (respectively second) moment of the displacement conditioned on the position. We obtain the parameters $\theta$ from the two equations
\begin{align}
F^{KM}(q) &=\left\langle (q_{i+1}-q_i)|q\right\rangle / \Delta t,\\
D^{KM}(q) &= \left\langle (q_{i+1}-q_i -F^{KM}(q_i) \Delta t  )^2 | q\right\rangle / \Delta t.
\end{align}


It is worth noticing that the Kramers-Moyal procedure would leads to the correct parameters in the limit of a small timestep $\Delta t$ and a sufficient number of trajectories passing at position $q$. In which case, it becomes equivalent to a maximum likelihood estimation using Euler discretization of the propagator. It thus provides a good starting guess for the maximum likelihood estimator.



## Parallel computation 

One key motivation behind the writing of folie was performance.
Likelihood calculation scales linearly with the data size, and with the model complexity, making optimization potentially slow. Furthermore, Langevin optimization can be integrated in a scheme for optimizing collextive variables [@mouaffac2023], in which case a large number of model optimizations must be performed before collective variable optimization converges.
When running folie in a shared-memory multiprocessor environment, likelihood computation is performed in a data-parallel way over the projected simulation trajectories.


# Practical use

## Usage workflow

![Typical workflow in folie. This basic workflow is illustrated in this [example script in the repository](https://github.com/langevinmodel/folie/blob/main/examples/plot_example.py). Several options are available for the model definition and the model estimation. \label{fig:worklow}](workflow.png)




# Perspectives

Further developments are in progress, in particular more widely applicable forms of dynamics, namely underdamped and generalized Langevin dynamics.
Thanks to the modular design of FOLIE, these will integrate seamlessly into the workflow.


# Acknowledgements

We are indebted to Fabio Pietrucci for spearheading the scientific effort that led us to develop FOLIE.
We acknowledge stimulating discussions with Arthur France-Lanord, David Girardier, Léo Hallegot, Léon Huet, and Line Mouaffac.

# References


@misc{GitHub_optLE,
  howpublished = "\url{https://github.com/physix-repo/optLE}"
}

@misc{GitHub_colvars,
  howpublished = "\url{https://github.com/Colvars/colvars}"
}

@article{PalacioRodriguez2022,
author = {Palacio-Rodriguez, Karen and Pietrucci, Fabio},
title = {Free Energy Landscapes, Diffusion Coefficients, and Kinetic Rates from Transition Paths},
journal = {Journal of Chemical Theory and Computation},
volume = {18},
number = {8},
pages = {4639-4648},
year = {2022},
doi = {10.1021/acs.jctc.2c00324},
note ={PMID: 35899416},
URL = {https://doi.org/10.1021/acs.jctc.2c00324},
eprint = {https://doi.org/10.1021/acs.jctc.2c00324}
}


@article{Marchi1999,
author = {Marchi, Massimo  and Ballone, Pietro },
title = {Adiabatic bias molecular dynamics: A method to navigate the conformational space of complex molecular systems},
journal = {The Journal of Chemical Physics},
volume = {110},
number = {8},
pages = {3697-3702},
year = {1999},
doi = {10.1063/1.478259},
URL = {https://doi.org/10.1063/1.478259},
eprint = {https://doi.org/10.1063/1.478259}
}


@article{Grubmueller1996,
author = {Helmut Grubmüller  and Berthold Heymann  and Paul Tavan },
title = {Ligand Binding: Molecular Mechanics Calculation of the Streptavidin-Biotin Rupture Force},
journal = {Science},
volume = {271},
number = {5251},
pages = {997-999},
year = {1996},
doi = {10.1126/science.271.5251.997},
URL = {https://www.science.org/doi/abs/10.1126/science.271.5251.997},
}


@Article{Wolf2020,
author={Wolf, Steffen
and Lickert, Benjamin
and Bray, Simon
and Stock, Gerhard},
title={Multisecond ligand dissociation dynamics from atomistic simulations},
journal={Nature Communications},
year={2020},
month={Jun},
day={10},
volume={11},
number={1},
pages={2918},
issn={2041-1723},
doi={10.1038/s41467-020-16655-1},
url={https://doi.org/10.1038/s41467-020-16655-1}
}

@article{Jarzynski1997,
  title = {Nonequilibrium Equality for Free Energy Differences},
  author = {Jarzynski, C.},
  journal = {Phys. Rev. Lett.},
  volume = {78},
  issue = {14},
  pages = {2690--2693},
  numpages = {0},
  year = {1997},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.78.2690},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.78.2690}
}

@book{Risken1996,
author="Risken, Hannes",
title="Fokker-Planck Equation",
bookTitle="The Fokker-Planck Equation: Methods of Solution and Applications",
year="1996",
publisher="Springer Berlin Heidelberg",
address="Berlin, Heidelberg",
doi="https://doi.org/10.1007/978-3-642-61544-3",
}

@article{hoffmann2021deeptime,
  title={Deeptime: a Python library for machine learning dynamical models from time series data},
  author={Hoffmann, Moritz and Scherer, Martin Konrad and Hempel, Tim and Mardt, Andreas and de Silva, Brian and Husic, Brooke Elena and Klus, Stefan and Wu, Hao and Kutz, J Nathan and Brunton, Steven and Noé, Frank},
  journal={Machine Learning: Science and Technology},
  year={2021},
  publisher={IOP Publishing}
}

@article{kirkby2024pymle, title={pymle: A Python Package for Maximum Likelihood Estimation and Simulation of Stochastic Differential Equations}, author={Kirkby, Justin and Nguyen, Dang and Nguyen, Duy and Nguyen, Nhu N}, journal={Journal of Statistical Software, Forthcoming}, year={2025} }

@article{mouaffac2023,
  title = {Optimal {{Reaction Coordinates}} and {{Kinetic Rates}} from the {{Projected Dynamics}} of {{Transition Paths}}},
  author = {Mouaffac, Line and {Palacio-Rodriguez}, Karen and Pietrucci, Fabio},
  year = {2023},
  month = sep,
  journal = {Journal of Chemical Theory and Computation},
  volume = {19},
  number = {17},
  pages = {5701--5711},
  publisher = {American Chemical Society},
  issn = {1549-9618},
  doi = {10.1021/acs.jctc.3c00158},
  urldate = {2024-02-04},
  abstract = {Finding optimal reaction coordinates and predicting accurate kinetic rates for activated processes are two of the foremost challenges of molecular simulations. We introduce an algorithm that tackles the two problems at once: starting from a limited number of reactive molecular dynamics trajectories (transition paths), we automatically generate with a Monte Carlo approach a sequence of different reaction coordinates that progressively reduce the kinetic rate of their projected effective dynamics. Based on a variational principle, the minimal rate accurately approximates the exact one, and it corresponds to the optimal reaction coordinate. After benchmarking the method on an analytic double-well system, we apply it to complex atomistic systems: the interaction of carbon nanoparticles of different sizes in water.},
}

@article{girardier2023,
  title = {Inferring Free-Energy Barriers and Kinetic Rates from Molecular Dynamics via Underdamped {{Langevin}} Models},
  author = {Girardier, David Daniel and Vroylandt, Hadrien and Bonella, Sara and Pietrucci, Fabio},
  year = {2023},
  month = oct,
  journal = {The Journal of Chemical Physics},
  volume = {159},
  number = {16},
  pages = {164111},
  issn = {0021-9606},
  doi = {10.1063/5.0169050},
  urldate = {2024-02-05},
}

@article{vroylandt2022a,
  title = {On the Derivation of the Generalized {{Langevin}} Equation and the Fluctuation-Dissipation Theorem},
  author = {Vroylandt, Hadrien},
  year = {2022},
  month = dec,
  journal = {Europhysics Letters},
  volume = {140},
  number = {6},
  pages = {62003},
  publisher = {{EDP Sciences, IOP Publishing and Societ{\`a} Italiana di Fisica}},
  issn = {0295-5075},
  doi = {10.1209/0295-5075/acab7d},
  urldate = {2022-12-29},
}

@book{iacus2008a,
  title = {Simulation and {{Inference}} for {{Stochastic Differential Equations}}: {{With R Examples}}},
  shorttitle = {Simulation and {{Inference}} for {{Stochastic Differential Equations}}},
  author = {Iacus, Stefano M.},
  year = {2008},
  series = {Springer {{Series}} in {{Statistics}}},
  volume = {1},
  publisher = {Springer},
  address = {New York, NY},
  doi = {10.1007/978-0-387-75839-8},
  urldate = {2023-07-20},
  isbn = {978-0-387-75838-1 978-0-387-75839-8},
  langid = {english},
  keywords = {compuational statistics,inference for stochastic processes,Information,Likelihood,numerical methods,quantitative finance,Simulation,simulation methods,stochastic differential equations,stochastic process,Stochastic Processes,time series analysis},
  file = {/home/vroylan241/Zotero/storage/RF2RXS2S/Iacus - 2008 - Simulation and Inference for Stochastic Differenti.pdf}
}
