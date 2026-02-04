---
title: 'FOLIE: Finding Optimal Langevin Inferred Equations'
tags:
  - Python
  - molecular dynamics
  - Langevin dynamics
  - effective dynamics
authors:
  - name: Hadrien Vroylandt
    orcid: 0000-0002-2443-5901
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Daniele Bersano
    affiliation: 3
  - name: Jérôme Hénin
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3


affiliations:
 - name: Université Caen Normandie, ENSICAEN, CNRS, Normandie Univ, GREYC UMR 6072, 14000 Caen, France
   index: 1
   ror: 051kpcy16
 - name: Sorbonne Université, Institut des sciences du calcul et des données, ISCD - F-75005 Paris, France
   index: 2
   ror: 00y16rm59
 - name: Université Paris Cité, CNRS, Laboratoire de Biochimie Théorique UPR 9080, 75005, Paris, France
   index: 3
   ror: 00nvjgv40
date: 23 Mai 2025
bibliography: paper.bib

---

# Summary


This paper introduces FOLIE (Finding Optimal Langevin Inferred Equations), a versatile Python library designed to facilitate the analysis of high-dimensional molecular simulation trajectories in terms of low-dimensional dynamics. FOLIE enables scientists to fit low-dimensional stochastic differential equations (SDEs) to projected high-dimensional data, thereby extracting maximum dynamical insight from the simulations. Key features of FOLIE include robust estimation techniques, comprehensive analysis tools, and simulation capabilities to create synthetic datasets. Its highly modular architecture allows for the implementation of diverse SDEs, time discretization methods, and energy landscapes. By leveraging FOLIE, researchers can effectively extrapolate low-dimensional kinetics from limited simulation data, enhancing their ability to understand and predict complex molecular dynamics.


# Statement of need

Quantitative predictions of rare events in molecular and materials simulations suffer from the curse of dimensionality and the prohibitive cost of simulating relevant time scales, starting from microscopic time steps (on the order of $10^{-15} s$).
Such rare events include chemical reactions in chemistry, drug unbinding in pharmacology, and phase transitions in materials.
A strategy to make such predictions tractable is to construct intermediate, low-dimensional kinetic models of the time evolution of the system.
Such low-dimensional models are constructed based on a projection of the full dynamics onto a reduced set of collective variables. This set must fulfill two constraints: describe the process of interest, and be informative enough to capture the long-time dynamics of this process.

# State of the field

FOLIE is designed to allow easy and efficient inference of such models from projected molecular simulations. There exits several software packages performing related tasks, but FOLIE differ mainly by its flexibility in the description of the energy landscape and its modular construction for the estimation task.
[DeepTime](https://deeptime-ml.github.io) (previously pyEmma) [@hoffmann2021deeptime] is an equivalent for discrete Markovian processes. [pymle](https://github.com/jkirkby3/pymle) fits continuous SDEs but imposes restraints on the underlying energy landscapes, which are better suited to econometrics and financial markets [@kirkby2024pymle].
[OptLE](https://github.com/physix-repo/optle) [@PalacioRodriguez2022] is an experimental Fortran package that inspired this work. It focuses on method development and was not designed for flexibility or scalability.
[pyOptLE](https://github.com/jhenin/pyOptLE) was our first attempt at improving scalability, with a limited scope.
[StochasticForceInference](https://github.com/ronceray/StochasticForceInference) and [UnderdampedLangevinInference](https://github.com/ronceray/UnderdampedLangevinInference) from the Ronceray group perform the task of fitting continuous SDE, with a focus on biological applications and less flexibility in the underlying energy landscapes.


# Theoretical background


## Langevin Models

Several types of Langevin equations may be relevant for describing projected dynamics[@PalacioRodriguez2022,@girardier2023].
Projecting high-dimensional dynamics onto a collective variable $q$ leads to the generalized Langevin equation[@vroylandt2022a] that features a memory kernel. Assuming that the timescale of evolution of $q$ is slow with respect to its environment gives the memory-less (Markovian) Standard Langevin equation. If the inertial effect are quickly damped, a further approximation can be made, leading to the Overdamped Langevin Equation
\begin{equation}
\dot{q}= -\beta D(q)\frac{\partial A(q)}{\partial q}+ \frac{\partial D(q)}{\partial q} + \sqrt{2D(q)}\eta(t)
\end{equation}
where, using $\beta = \frac{1}{k_BT}$, $D(q)$ is a diffusion profile, the effective free energy surface is $A(q) = -k_BT \log (\rho_{eq}(q))$ where $\rho_{eq}$ is the invariant distribution of the dynamics and $\eta(t)$ is a standard Gaussian noise.
This is a first order differential equation, simplifying the mathematical structure.
The current focus of the library is the overdamped case.


## Kinetic model optimization by a maximum-likelihood approach

In order to construct the optimal Langevin model in low dimension, we define $\mathcal{L}( \vec{q}|\theta)$, the likelihood of observing the trajectory data $\vec{q}$ if they were generated by a Langevin model parameterized by $\theta$. Here $\theta$ represents both the drift $F(q)=-\beta D(q)\frac{\partial A(q)}{\partial q}+ \frac{\partial D(q)}{\partial q}$ and position-dependent diffusion $D(q)$.
$\theta$ is optimized maximizing the likelihood
\begin{equation}
    \hat{\theta}= \underset{\theta}{\mathrm{argmax}} \; \mathcal{L}(\theta).
\end{equation}

In the overdamped case, due to markovianity, the likelihood is the product of transition probabilities $p_\theta(q_{i+1},t_{i+1}|q_{i},t_{i})$ between consecutive points [@PalacioRodriguez2022]:
\begin{equation}
\mathcal{L}(\vec{q}|\theta ) = \prod_{i=0}^{N-1} p_\theta(q_{i+1},t_{i+1}|q_{i},t_{i})
\end{equation}
and the log-likelihood is :
\begin{equation} \label{eq:log_likelihood_general}
    \log{\mathcal{L}}(\vec{q}|\theta) = \sum_{i=0}^{N-1} \log \left[ p_\theta(q_{i+1},t_{i+1}|q_{i},t_{i}) \right]
\end{equation}

The precise form of the transition density depends on a specific time discretization (closely related to the choice of an integrator) of the continuous SDE, several of which are implemented in FOLIE.

# Software Design

## Features

The library follows a modular structure, allowing users to assemble components into Python scripts suited to their needs.

- **Model of Overdamped Langevin Dynamics**
    Implements several forms of Overdamped Langevin equations, along with particular cases ($\texttt{BrownianMotion}$, $\texttt{OrnsteinUhlenbeck}$). The underdamped case is still under development.

- **Force and diffusion coefficient functions**
    Defining the dynamical model requires specifying the drift and space-dependent diffusion coefficient. To that effect, the $\texttt{Function}$ class offers functional forms such as polynomials, splines, etc.

- **Transition densities**
    Several approximations for the propagator are implemented[@iacus2008a]. These probability densities are later fed to the $\texttt{Likelihood}$ estimator object, which optimizes the drift and diffusion parameters.

 - **Estimation**
    The $\texttt{LikelihoodEstimator}$ class performs parameter estimation by maximizing the likelihood of observed trajectories using [scipy.optimize.minimize()](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.optimize.minimize.html) on the negative log-likelihood function of eq. (\ref{eq:log_likelihood_general}).


- **Simulation**
    The FOLIE module can also generate simulated trajectories, which is useful for creating synthetic data to be used when developing methods.


## Initial guess

Optimization of the likelihood requires an initial guess for the drift and diffusion parameter. In FOLIE, Kramers-Moyal estimation provides such a guess[@Risken1996]. This yields accurate parameters for small timesteps $\Delta t$ and sufficient trajectories, being then equivalent to maximum-likelihood estimation with Euler discretization. This makes it a strong initial estimate for maximum-likelihood approaches.

## Parallel computation

Likelihood calculation scales linearly with data size and model complexity, making optimization potentially slow. Furthermore, Langevin optimization can be integrated in a scheme for optimizing collective variables [@mouaffac2023], in which case a large number of model optimizations must be performed before collective variable optimization converges. When running FOLIE in a shared-memory multiprocessor environment, likelihood computation is performed in a data-parallel way over the projected simulation trajectories.


# Practical use

## Usage workflow

![Typical workflow in FOLIE. This basic workflow is illustrated in this [example script in the repository](https://github.com/langevinmodel/folie/blob/main/examples/plot_example.py). Several options are available for the model definition and estimation. \label{fig:worklow}](workflow.png)



# Research Impact Statement


## Perspective
Further developments are in progress, in particular the more widely applicable underdamped and generalized Langevin dynamics.
Thanks to the modular design of FOLIE, these will integrate seamlessly into the workflow.



# AI usage disclosure

No AI-tools were used in the software development or the documention. AI tools (ChatGPT) were used to reduce the size of this paper.

# Acknowledgements

We are indebted to Fabio Pietrucci for spearheading the scientific effort that led us to develop FOLIE.
We acknowledge stimulating discussions with Arthur France-Lanord, David Girardier, Léo Hallegot, Léon Huet, and Line Mouaffac.

# References


