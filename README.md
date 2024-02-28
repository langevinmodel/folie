# folie
[![codecov](https://codecov.io/gh/langevinmodel/folie/graph/badge.svg?token=V6R4HF2FBJ)](https://codecov.io/gh/langevinmodel/folie)
![CI](https://github.com/langevinmodel/folie/actions/workflows/CI.yml/badge.svg)

 Finding Optimal Langevin Inferred Equations


## About

folie intends to provide a simple to use module for inference of Langevin equations from trajectories of collectives variables. Please refer to the documentation for a more complete description.

folie is currently WIP so feel free to provide feedback or potential bugs in the GitHub issue tracker.


folie is partly based on [pymle](https://github.com/jkirkby3/pymle), [pyoptLe](https://github.com/jhenin/pyOptLE) and  [deeptime](https://github.com/deeptime-ml/deeptime). In particuler, the interface of folie should be compatible with [deeptime](https://github.com/deeptime-ml/deeptime).

## Installation

```
pip install git+https://github.com/langevinmodel/folie.git
```

## Examples

Various examples can be found in the `examples` folder on how to perform inference using folie.


## Build the documentation

The documention is available on the [Github pages](https://langevinmodel.github.io/folie/). It can also be compiled by first installing the extra dependencies using pip and then compile it.


```
git clone https://github.com/langevinmodel/folie.git
cd folie
pip install .[docs]
cd docs
make html
```

## Help and comments

Please use the github issue of this repository.
