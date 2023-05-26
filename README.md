[![Build Status](https://travis-ci.org/mppmu/qmc.svg?branch=master)](https://travis-ci.org/mppmu/qmc)
[![Coverage Status](https://coveralls.io/repos/github/mppmu/qmc/badge.svg?branch=master)](https://coveralls.io/github/mppmu/qmc?branch=master)

[The latest release of the single header can be downloaded directly using this link.](https://github.com/mppmu/qmc/releases/download/v1.1.0/qmc.hpp)

# qmc

A Quasi-Monte-Carlo (QMC) integrator library with NVIDIA CUDA support.

The library can be used to integrate multi-dimensional real or complex functions numerically. Multi-threading is supported via the C++17 threading library and multiple CUDA compatible accelerators are supported. A variance reduction procedure based on fitting a smooth function to the inverse cumulative distribution function of the integrand dimension-by-dimension is also implemented.

To read more about the library see [our publication](https://arxiv.org/abs/1811.11720).

## Installation

Prerequisites:
* A C++17 compatible C++ compiler.
* (Optional GPU support)  A CUDA compatible compiler (typically `nvcc`).
* (Optional GPU support) CUDA compatible hardware with Compute Capability 3.0 or greater.

The qmc library is header only. Simply put the single header file somewhere reachable from your project or directly into your project tree itself then `#include "qmc.hpp"` in your project.

## Usage

Example: Integrate x0*x1*x2 over the unit hypercube
```cpp
#include <iostream>
#include "qmc.hpp"

struct my_functor_t {
  const unsigned long long int number_of_integration_variables = 3;
#ifdef __CUDACC__
  __host__ __device__
#endif
  double operator()(double* x) const
  {
    return x[0]*x[1]*x[2];
  }
} my_functor;

int main() {

  const unsigned int MAXVAR = 3; // Maximum number of integration variables of integrand

  integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
  integrators::result<double> result = integrator.integrate(my_functor);
  std::cout << "integral = " << result.integral << std::endl;
  std::cout << "error    = " << result.error    << std::endl;

  return 0;
}
```

Compile without GPU support:
```shell
$ c++ -std=c++17 -pthread -I../src 1_minimal_demo.cpp -o 1_minimal_demo.out -lgsl -lgslcblas
```

Compute with GPU support:
```shell
$ nvcc -arch=<arch> -std=c++17 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 1_minimal_demo.cpp -o 1_minimal_demo.out -lgsl -lgslcblas
```
where `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).

Output:
```shell
integral = 0.125
error    = 5.43058e-11
```

For further examples see the [examples folder](examples).

## API Documentation

The Qmc class has 7 template parameters:
* `T` the return type of the  function to be integrated (assumed to be a real or complex floating point type)
* `D` the argument type of the function to be integrated (assumed to be a floating point type)
* `M` the maximum number of integration variables of any integrand that will be passed to the integrator
* `P` an integral transform to be applied to the integrand before integration
* `F` a function to be fitted to the inverse cumulative distribution function of the integrand in each dimension, used to reduce the variance of the integrand (default: `fitfunctions::None::template type`)
* `G` a C++17 style pseudo-random number engine (default: `std::mt19937_64`)
* `H` a C++17 style uniform real distribution (default: `std::uniform_real_distribution<D>`)

Internally, unsigned integers are assumed to be of type `U = unsigned long long int`.

Typically the return type `T` and argument type `D` are set to type `double` (for real numbers), `std::complex<double>` (for complex numbers on the CPU only) or `thrust::complex<double>`  (for complex numbers on the GPU and CPU). In principle, the qmc library supports integrating other floating point types (e.g. quadruple precision, arbitrary precision, etc), though they must be compatible with the relevant STL library functions or provide compatible overloads. 

To integrate alternative floating point types, first include the header(s) defining the new type into your project and set the template arguments of the Qmc class `T` and `D` to your type. The following standard library functions must be compatible with your type or a compatible overload must be provided:
* `sqrt`, `abs`, `modf`, `pow`
* `std::max`, `std::min`

If your type is not intended to represent a real or complex type number then you may also need to overload functions required for calculating the error resulting from the numerical integration, see the files `src/overloads/real.hpp` and `src/overloads/complex.hpp`. 

Example `9_boost_minimal_demo` demonstrates how to instantiate the qmc with a non-standard type (`boost::multiprecision::cpp_bin_float_quad`), to compile this example you will need the `boost` library available on your system.

### Public fields

---

`Logger logger`

A wrapped `std::ostream` object to which log output from the library is written. 

To write the text output of the library to a particular file, first `#include <fstream>`, create a `std::ofstream` instance pointing to your file then set the `logger` of the integrator to the `std::ofstream`. For example to output very detailed output to the file `myoutput.log`:
```cpp
std::ofstream out_file("myoutput.log");
integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
integrator.verbosity=3;
integrator.logger = out_file;
```

Default: `std::cout`.

---

`G randomgenerator`

A C++17 style pseudo-random number engine. 

The seed of the pseudo-random number engine can be changed via the `seed` member function of the pseudo-random number engine.
For total reproducability you may also want to set `cputhreads = 1`  and `devices = {-1}` which disables multi-threading, this helps to ensure that the floating point operations are done in the same order each time the code is run.
For example:
```cpp
integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
integrator.randomgenerator.seed(1) // seed = 1
integrator.cputhreads = 1; // no multi-threading
integrator.devices = {-1}; // cpu only
```

Default: `std::mt19937_64` seeded with a call to `std::random_device`.

---

`U minn`

The minimum lattice size that should be used for integration. If a lattice of the requested size is not available then `n` will be the size of the next available lattice with at least `minn` points. 

Default: `8191`.

---

`U minm`

The minimum number of random shifts of the lattice `m` that should be used to estimate the error of the result. Typically 10 to 50. 

Default: `32`.

---

`D epsrel`

The relative error that the qmc should attempt to achieve. 

Default: `0.01`.

---

`D epsabs`

The absolute error that the qmc should attempt to achieve. For real types the integrator tries to find an estimate `E` for the integral `I` which fulfills  `|E-I| <= max(epsabs, epsrel*I)`. For complex types the goal is controlled by the `errormode` setting.

Default: `1e-7`.

---

`U maxeval`

The (approximate) maximum number of function evaluations that should be performed while integrating. The actual number of function evaluations can be slightly larger if there is not a suitably sized lattice available. 

Default: `1000000`.

---

`U maxnperpackage`

Maximum number of points to compute per thread per work package. 

Default: `1`.

---

`U maxmperpackage`

Maximum number of shifts to compute per thread per work package. 

Default: `1024`.

---

`ErrorMode errormode`

Controls the error goal that the library attempts to achieve when the integrand return type is a complex type. For real types the `errormode` setting is ignored.

Possible values:
*  `all` - try to find an estimate `E` for the integral `I` which fulfills  `|E-I| <= max(epsabs, epsrel*I)` for each component (real and imaginary) separately,
*  `largest` - try to find an estimate `E` for the integral `I` such that `max( |Re[E]-Re[I]|, |Im[E]-Im[I]| ) <= max( epsabs, epsrel*max( |Re[I]|,|Im[I]| ) )` , i.e. to achieve either the `epsabs` error goal or that the largest error is smaller than `epsrel` times the value of the largest component (either real or imaginary).

Default: `all`.

---

`U cputhreads`

The number of CPU threads that should be used to evaluate the integrand function. If GPUs are used 1 additional CPU thread per device will be launched for communicating with the device. 

Default: `std::thread::hardware_concurrency()`.

---

`U cudablocks`

The number of blocks to be launched on each CUDA device. 

Default: (determined at run time).

---

`U cudathreadsperblock`

The number of threads per block to be launched on each CUDA device. CUDA kernels launched by the qmc library have the execution configuration `<<< cudablocks, cudathreadsperblock >>>`. For more information on how to optimally configure these parameters for your hardware and/or integral refer to the NVIDIA guidelines. 

Default: (determined at run time).

---

`std::set<int> devices`

A set of devices on which the integrand function should be evaluated. The device id `-1` represents all CPUs present on the system, the field `cputhreads` can be used to control the number of CPU threads spawned. The indices `0,1,...` are device ids of CUDA devices present on the system. 

Default: `{-1,0,1,...,nd}` where `nd` is the number of CUDA devices detected on the system.

---

`std::map<U,std::vector<U>> generatingvectors`

A map of available generating vectors which can be used to generate a lattice. The implemented QMC algorithm requires that the generating vectors be generated with a prime lattice size. By default the library uses generating vectors with 100 components, thus it supports integration of functions with up to 100 dimensions.
The default generating vectors have been generated with lattice size chosen as the next prime number above `(110/100)^i*1020` for `i` between `0` and `152`, additionally the lattice `2^31-1` (`INT_MAX` for `int32`) is included. 

Default: `cbcpt_dn1_100()`.

---

`U lattice_candidates`
If `lattice_candidates>0`, the list of generating vectors is extended using the [median quasi-Monte Carlo rules](https://arxiv.org/abs/2201.09413), 
using the given number of candidate generating vectors.
Can be used together with `generatingevectors=none()` to always use the median QMC rule.

Default: `11`.

---

`bool keep_lattices`

If set to `true`, generating vectors constructed using the median QMC rules are kept in `generatingvectors` for subsequent integrations.

Default: `false`.

---

`U verbosity`

Possible values: `0,1,2,3`. Controls the verbosity of the output to `logger` of the qmc library.
* `0` - no output,
* `1` - key status updates and statistics,
* `2` - detailed output, useful for debugging,
* `3` - very detailed output, useful for debugging.

Default: `0`.

---

`bool batching`

If set to `true`, attempts to compute batches of points on the cpu. This allows the user to make better use of SIMD instructions on their hardware.

If the user provides it, on the cpu the integrator will use the call operator:  
```cpp
void operator()(double* x, double* res, const U batchsize) const
```
This call operator should be ready to accept up to `maxnperpackage` points.

The parameters are:
* `x`  - a one-dimensional array first containing coordinates of point number `0`, then point number `1` and so on,
* `res` - the array of results,
* `batchsize` - the number of points passed to the function.

Dafault: `false`.

---


`U evaluateminn`

The minimum lattice size that should be used by the `evaluate` function to evaluate the integrand, if variance reduction is enabled these points are used for fitting the inverse cumulative distribution function. If a lattice of the requested size is not available then `n` will be the size of the next available lattice with at least `evaluateminn` points. 

Default: `100000`.

---

`size_t fitstepsize`

Controls the number of points included in the fit used for variance reduction. A step size of `x` includes (after sorting by value) every `x`th point in the fit.

Default: `10`.

---

`size_t fitmaxiter`

See `maxiter` in the non-linear least-squares fitting GSL documentation.

Default: `40`.

---

`double fitxtol`

See `xtol` in the non-linear least-squares fitting GSL documentation.

Default: `3e-3`.

---

`double fitgtol`

See `gtol`  in the non-linear least-squares fitting GSL documentation.

Default: `1e-8`.

---

`double fitftol`

See `ftol`  in the non-linear least-squares fitting GSL documentation.

Default: `1e-8`.

---

`gsl_multifit_nlinear_parameters fitparametersgsl`

See `gsl_multifit_nlinear_parameters` in the non-linear least-squares fitting GSL documentation.

Default: `{}`.

### Public Member Functions

`U get_next_n(U preferred_n)`

Returns the lattice size `n` of the lattice in `generatingvectors` that is greater than or equal to `preferred_n`. This represents the size of the lattice that would be used for integration if `minn` was set to `preferred_n`.

---

`template <typename I> result<T,U> integrate(I& func)`

Integrates the functor `func`. The result is returned in a `result` struct with the following members:
* `T integral` - the result of the integral
* `T error` - the estimated absolute error of the result
* `U n` - the size of the largest lattice used during integration
* `U m` - the number of shifts of the largest lattice used during integration
* `U iterations` - the number of iterations used during integration
* `U evaluations` - the total number of function evaluations during integration

The functor `func` must define its dimension as a public member variable `number_of_integration_variables`.

Calls: `get_next_n`.

---

`template <typename I> samples<T,D> evaluate(I& func)`

Evaluates the functor `func` on a lattice of size greater than or equal to `evaluateminn`.  The samples are returned in a `samples` struct with the following members:
* `std::vector<U> z` - the generating vector of the lattice used to produce the samples
* `std::vector<D> d` - the random shift vector used to produce the samples
* `std::vector<T> r` - the values of the integrand at each randomly shifted lattice point
* `U n` - the size of the lattice used to produce the samples
* `D get_x(const U sample_index, const U integration_variable_index)` - a function which returns the argument (specified by `integration_variable_index`) used to evaluate the integrand for a specific sample (specified by `sample_index`).

The functor `func` must define its dimension as a public member variable `number_of_integration_variables`.

Calls: `get_next_n`.

---

`template <typename I> typename F<I,D,M>::transform_t fit(I& func)`

Fits a function (specified by the type `F` of the integrator) to the inverse cumulative distribution function of the integrand dimension-by-dimension and returns a functor representing the new integrand after this variance reduction procedure.

The functor `func` must define its dimension as a public member variable `number_of_integration_variables`.

Calls: `get_next_n`, `evaluate`.

### Generating Vectors

The following generating vectors are distributed with the qmc:

| Name | Max. Dimension | Description | Lattice Sizes |
| --- | --- | --- | --- |
| `cbcpt_dn1_100` | 100 | Computed using Dirk Nuyens' [fastrank1pt.m tool](https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc) | 1021 - 2147483647 |
| `cbcpt_dn2_6`     | 6     | Computed using Dirk Nuyens' [fastrank1pt.m tool](https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc) | 65521 - 2499623531 |
| `cbcpt_cfftw1_6` | 6     | Computed using a custom CBC tool based on FFTW | 2500000001 - 15173222401 |
| `cbcpt_cfftw2_10` | 10     | Computed using a custom CBC tool based on FFTW | 2147483659 - 68719476767 |
| `none` | inf     | Empty list of generating vectors, to be filled using median Qmc rule | arbitrary |

The above generating vectors are produced for Korobov spaces with smoothness `alpha=2` using:
* Kernel `omega(x)=2 pi^2 (x^2 - x + 1/6)`,
* Weights `gamma_i = 1/s` for `i = 1, ..., s`,
* Parameters `beta_i = 1` for `i = 1, ..., s`.

The generating vectors used by the qmc can be selected by setting the integrator's `generatingvectors` member variable. Example (assuming an integrator instance named `integrator`):
```cpp
integrator.generatingvectors = integrators::generatingvectors::cbcpt_dn2_6();
```

If you prefer to use custom generating vectors and/or 100 dimensions and/or 15173222401 lattice points is not enough, you can supply your own generating vectors. Compute your generating vectors using another tool then put them into a map and set `generatingvectors`. 

If you prefer to use custom generating vectors and/or 100 dimensions and/or 15173222401 lattice points is not enough, you can supply your own generating vectors. Compute your generating vectors using another tool then put them into a map and set `generatingvectors`. For example, to instruct the qmc to use only two generating vectors (`z = (1,3)` for `n=7` and `z = (1,7)`  for `n=11`) the `generatingvectors` map would be set as follows:
```cpp
std::map<unsigned long long int,std::vector<unsigned long long int>> my_generating_vectors = { {7, {1,3}}, {11, {1,7}} };
integrators::Qmc<double,double,10> integrator;
integrator.generatingvectors = my_generating_vectors;
```
If you think your generating vectors will be widely useful for other people then please let us know! With your permission we may include them in the code by default.

### Integral Transforms

The following integral transforms are distributed with the qmc:

| Name | Description |
| --- | --- |
| `Korobov<r_0,r_1>` | A polynomial integral transform with weight ∝ x^r_0 * (1-x)^r_1   |
| `Korobov<r>` | A polynomial integral transform with weight ∝ x^r * (1-x)^r   |
| `Sidi<r>` | A trigonometric integral transform with weight ∝ sin^r(pi*x) | 
| `Baker` | The baker's transformation, phi(x) = 1 - abs(2x-1)  |
| `None` | The trivial transform, phi(x) = x  |

The integral transform used by the qmc can be selected when constructing the qmc.
Example (assuming a real type integrator instance named `integrator`):
```cpp
integrators::Qmc<double,double,10,integrators::transforms::Korobov<5,3>::type> integrator;
```
instantiates an integrator which applies a weight `(r_0=5,r_1=3)` Korobov transform to the integrand before integration.

### Fit Functions

| Name | Description |
| --- | --- |
| `PolySingular` | A 3rd order polynomial with two additional `1/(p-x)` terms, `f(x) = \|p_2\|*(x*(p_0-1))/(p_0-x) + \|p_3\|*(x*(p_1-1))/(p_1-x)  + x*(p_4+x*(p_5+x*(1-\|p_2\|-\|p_3\|-p_4-p_5)))` |
| `None` | No fit is performed |

The fit function used by the qmc can be selected when constructing the qmc. These functions are used to approximate the inverse cumulative distribution function of the integrand dimension-by-dimension.
Example (assuming a real type integrator instance named `integrator`):
```cpp
integrators::Qmc<double,double,10,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> integrator;
```
instantiates an integrator which reduces the variance of the integrand by fitting a `PolySingular` type function before integration.

## Authors

* Sophia Borowka (@sborowka)
* Gudrun Heinrich (@gudrunhe)
* Stephan Jahn (@jPhy)
* Stephen Jones (@spj101)
* Matthias Kerner (@KernerM)
* Johannes Schlenk (@j-schlenk)
