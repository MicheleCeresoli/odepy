# odepy
_An extremely fast propagator for Python, in Python._

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://micheleceresoli.github.io/odepy/)
[![Build Status](https://github.com/MicheleCeresoli/odepy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MicheleCeresoli/odepy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MicheleCeresoli/odepy/branch/main/graph/badge.svg?token=ECDAU1ZURX)](https://codecov.io/gh/MicheleCeresoli/odepy)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This is a small suite of **Ordinary Differential Equations (ODEs)** Runge-Kutta solvers for Python that leverages [Numba](https://numba.pydata.org/) to minimise integration times, outperforming well established libraries like [SciPy](https://scipy.org/). Additionally, odepy's integrator capability to be called within numba-decorated functions allows the optimisation and decoration of much larger portion of codes.

Each of the available Runge-Kutta solvers comes with a dedicated dense output interpolant: 
  1. **BS5** - Bogacki-Shampine 5/4 method (lazy 5th order interpolant).
  2. **DOP853** - Hairer's 8/5/3 adaptation of the Dormand-Prince method (7th order interpolant)
  1. **Vern7** - Verner's "Most Efficient" 7/6 method (lazy 6th order interpolant).
  2. **Vern8** - Verner's "Most Efficient" 8/7 method (lazy 7th order interpolant).  
  3. **Vern9** - Verner's "Most Efficient" 9/8 method (lazy 8th order interpolant).
