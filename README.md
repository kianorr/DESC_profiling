# DESC_profiling

# Need to change the goals

## Goals
1. To profile the current performance of DESC for the calculation of a three-dimensional plasma equilibrium using GPUs and CPUs.

2. To identify the most computationally expensive processes executed by JAX in DESC in terms of
memory and run time
    - singular_integral
    - zernike polynomials

3. To investigate optimizations into DESC based on the previously identified expensive processes
    - direct evaluation of zernike polynomials instead of jacobi polynomials

4. To investigate the effect of using GPU versus CPU for different functionalities of DESC
