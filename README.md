# odepy
_An extremely fast propagator for Python, in Python._

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://micheleceresoli.github.io/odepy/)
[![Build Status](https://github.com/MicheleCeresoli/odepy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MicheleCeresoli/odepy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MicheleCeresoli/odepy/branch/main/graph/badge.svg?token=ECDAU1ZURX)](https://codecov.io/gh/MicheleCeresoli/odepy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This is a small suite of **Ordinary Differential Equations (ODEs)** Runge-Kutta solvers for Python that leverages [Numba](https://numba.pydata.org/) to minimise integration times, outperforming well established libraries like [SciPy](https://scipy.org/). Additionally, odepy's integrator capability to be called within numba-decorated functions allows the optimisation and decoration of much larger portion of code.

## Installation 
You can install odepy from [PyPI](https://pypi.org/project/odepy/): 
```
pip install odepy
```
odepy runs on Python 3.7+ and only requires [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Numba](https://numba.pydata.org/). However, it cannot be installed on Python 3.11 until official support for Numba is released for that Python version.

## Features
This package has been designed to maximise performance and flexibility. This is achieved by optimising and extending some of Scipy's built-in features, including: 
 - H. A. Watts' initial step size selection [[1](https://www.sciencedirect.com/science/article/pii/0377042783900407)]
 - Horner's rule for dense output polynomial evaluation [[2](https://en.wikipedia.org/wiki/Horner%27s_method)]
 - In-place vector and matrices computations 

Advanced stepsize control can be enabled for all of odepy's solvers but DOP853 (which comes with its own custom error estimation and control method), including: 
 - **Integral Controller**: the standard proportional control algorithm, adopted by SciPy. 
 - **PI Controller**: second order proportional-integral controllers with improved stability properties [[3](https://link.springer.com/book/10.1007/978-3-642-05221-7)-[4](https://link.springer.com/book/10.1007/978-3-540-78862-1)]
 - **PID Controller**: further improves the stability and efficiency properties of PI Controllers [[5](https://linkinghub.elsevier.com/retrieve/pii/S0377042705001123)-[6](https://docs.sciml.ai/DiffEqDocs/dev/extras/timestepping/)]
 
For a list of all the available controllers, please refer to the [Stepsize control]() documentation.

Additionally, each of the available Runge-Kutta solvers comes with a dedicated dense output interpolant: 
  - **BS5** - Bogacki-Shampine 5/4 method (lazy 5th order interpolant).
  - **DOP853** - Hairer's 8/5/3 adaptation of the Dormand-Prince method (7th order interpolant)
  - **Vern7** - Verner's "Most Efficient" 7/6 method (lazy 6th order interpolant).
  - **Vern8** - Verner's "Most Efficient" 8/7 method (lazy 7th order interpolant).  
  - **Vern9** - Verner's "Most Efficient" 9/8 method (lazy 8th order interpolant).

For further details on these algorithms, [see the stable documentation]().

## Customisation 
You can easily extend odepy's built-in methods and stepsize controllers by defining your own Runge-Kutta tableau and set of control coefficients. Check out the [tutorials]() to start enjoying the full flexibility of this package.

## Documentation 
Please refer to the [documentation](https://micheleceresoli.github.io/odepy/) for additional information.

## Supporting 
The software was developed as part of academic research by [Michele Ceresoli](https://github.com/MicheleCeresoli) and [Andrea Pasquale](https://github.com/andreapasquale94). If you found this package useful, please consider starring the repository. 
