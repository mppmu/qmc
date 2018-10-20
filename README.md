# qmc

A Quasi-Monte-Carlo integrator library with Nvidia CUDA support.

The library can be used to integrate multi-dimensional real or complex functions numerically. Multi-threading is supported via the C++11 threading library and multiple CUDA compatible accelerators are supported. A variance reduction procedure based on fitting a smooth function to the inverse cumulative distribution function of the integrand dimension-by-dimension is also implemented.

## Installation (standalone)

The latest release of the single header can be downloaded directly using this link. - TODO - link

Prerequisites:
* A C++11 compatible C++ compiler.
* (Optional GPU support)  A CUDA compatible compiler (typically `nvcc` or `clang`).
* (Optional GPU support) CUDA compatible hardware with Compute Capability 3.0 or greater.

The qmc library is header only. Simply put the single header file somewhere reachable from your project or directly into your project tree itself then `#include "qmc.hpp"` in your project.

## Installation (with [pysecdec](https://github.com/mppmu/secdec))

The qmc integrator library is redistributed by the `pysecdec` project and will be automatically installed for use within `pysecdec` by following the usual `pysecdec` installation procedure. There is no need to follow the standalone installation procedure to use the qmc with `pysecdec`.

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
$ c++ -std=c++11 -pthread -I../src 1_minimal_demo.cpp -o 1_minimal_demo.out -lgsl -lgslcblas
```

Compute with GPU support:
```shell
$ nvcc -std=c++11 -x cu -I../src 1_minimal_demo.cpp -o 1_minimal_demo.out -lgsl -lgslcblas
```

Output:
```shell
integral = 0.125
error    = 5.43058e-11
```

For further examples see the [examples folder](examples).

## Implementation Details

TODO:
* What is a work package, how does this play with maxnperpackage, maxmperpackage
* How is the load balanced
* How is the error goal acheived
* When is memory allocated on the devices, where is the result computed?
* How does the error estimation work, error always just 1 lattice with shifts (so iterating over many lattices is suboptimal) 


## API Documentation

The Qmc class has 7 template parameters:
* `T` the return type of the  function to be integrated (assumed to be a floating point type)
* `D` the argument type of the function to be integrated (assumed to be a floating point type) 
* `M` the maximum number of integration variables of any integrand that will be passed to the integrator
* `P` an integral transform to be applied to the integrand before integration
* `F` a function to be fitted to the inverse cumulative distribution function of the integrand in each dimension, used to reduce the variance of the integrand
* `G` a C++11 style pseudo-random number engine (default: `std::mt19937_64`)
* `H` a C++11 style uniform real distribution (default: `std::uniform_real_distribution<D>`)

### Public fields

---

`Logger logger`

A wrapped `std::ostream` object to which log output from the library is written. 

Default: `std::cout`.

---

`G randomgenerator`

A C++11 style pseudo-random number engine. 

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

Controls the error goal that the library attempts to acheive when the integrand return type is a complex type. For real types the `errormode` setting is ignored.

Possible values:
*  `all` - try to find an estimate `E` for the integral `I` which fulfills  `|E-I| <= max(epsabs, epsrel*I)` for each component (real and imaginary) separately,
*  `largest` - try to find an estimate `E` for the integral `I` such that `max( |Re[E]-Re[I]|, |Im[E]-Im[I]| ) <= min( epsabs, epsrel*max( |Re[I]|,|Im[I]| ) )` , i.e. to achieve either the `epsabs` error goal or that the largest error is smaller than `epsrel` times the value of the largest component (either real or imaginary).

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

The number of threads per block to be launched on each CUDA device. CUDA kernels launched by the qmc library have the execution configuration `<<< cudablocks, cudathreadsperblock >>>`. For more information on how to optimally configure these parameters for your hardware and/or integral refer to the Nvidia guidelines. 

Default: (determined at run time).

---

`std::set<int> devices`

A set of devices on which the integrand function should be evaluated. The device id `-1` represents all CPUs present on the system, the field `cputhreads` can be used to control the number of CPU threads spawned. The indices `0,1,...` are device ids of CUDA devices present on the system. 

Default: `{-1,0,1,...,nd}` where `nd` is the number of CUDA devices detected on the system.

---

`std::map<U,std::vector<U>> generatingvectors`

A map of available generating vectors which can be used to generate a lattice. The implemented QMC algorithm requires that the generating vectors be generated with a prime lattice size. By default the library uses generating vectors with 100 components, thus it supports integration of functions with up to 100 dimensions.
The default generating vectors have been generated with lattice size chosen as the next prime number above `(110/100)^i*1020` for `i` between `0` and `152`, additionally the lattice `2^31-1` (`INT_MAX` for `int32`) is included. 

Default: `cbcpt_dn1_100<U>()`.

---

`U verbosity`

Possible values: `0,1,2,3`. Controls the verbosity of the output to `logger` of the qmc library.
* `0` - no output,
* `1` - key status updates and statistics,
* `2` - detailed output, useful for debugging,
* `3` - very detailed output, useful for debugging.

Default: `0`.

---

`U evaluateminn`

The minimum lattice size that should be used by the `evaluate` function to evaluate the integrand (used also for fitting the inverse cumulative distribution function). If a lattice of the requested size is not available then `n` will be the size of the next available lattice with at least `evaluateminn` points. 

Default: `100000`.

---

`int fitmaxiter`

See `maxiter` in the nonlinear least-squares fitting GSL documentation.

Default: `40`.

---

`double fitxtol`

See `xtol` in the nonlinear least-squares fitting GSL documentation.

Default: `3e-3`.

---

`double fitgtol`

See `gtol`  in the nonlinear least-squares fitting GSL documentation.

Default: `1e-4`.

---

`double fitftol`

See `ftol`  in the nonlinear least-squares fitting GSL documentation.

Default: `1e-8`.

---

`gsl_multifit_nlinear_parameters fitparametersgsl`

See `gsl_multifit_nlinear_parameters` nonlinear least-squares fitting GSL documentation.

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

---

`template <typename I> samples<T,D> evaluate(I& func)`

Evaluates the functor `func` on a lattice of size greater than or equal to `evaluateminn`.  The samples are returned in a `samples` struct with the following members:
*`std::vector<U> z` - the generating vector of the lattice used to produce the samples
*`std::vector<D> d` - the random shift vector used to produce the samples
*`std::vector<T> r` - the values of the integrand at each randomly shifted lattice point
*`U n` - the size of the lattice used to produce the samples
*`D get_x(const U sample_index, const U integration_variable_index)` - a function which returns the argument (specified by `integration_variable_index`) used to evaluate the integrand for a specific sample (specified by `sample_index`).

The functor `func` must define its dimension as a public member variable `number_of_integration_variables`.

---

`template <typename I> typename F<I,D,M>::transform_t fit(I& func)`

Fits a function (specified by the type `F` of the integrator) to the inverse cumulative distribution function of the integrand dimension-by-dimension and returns a functor representing the new integrand after this variance reduction procedure.

The functor `func` must define its dimension as a public member variable `number_of_integration_variables`.

### Generating Vectors

The following generating vectors are distributed with the qmc:

| Name | Max. Dimension | Description | Lattice Sizes |
| --- | --- | --- | --- |
| cbcpt_dn1_100 | 100 | Computed using Dirk Nuyens' [fastrank1pt.m tool](https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc) | 1021 - 2147483647 |
| cbcpt_dn2_6     | 6     | Computed using Dirk Nuyens' [fastrank1pt.m tool](https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc) | 65521 - 2499623531 |
| cbcpt_cfftw1_6 | 6     | Computed using a custom CBC tool based on FFTW | 2500000001 - 15173222401 |

The generating vectors used by the qmc can be selected by setting the integrator's `generatingvectors` member variable. Example (assuming an integrator instance named `integrator`):
```cpp
integrator.generatingvectors = integrators::generatingvectors::cbcpt_dn2_6();
```
### Integral Transforms

The following integral transforms are distributed with the qmc:

| Name | Description |
| --- | --- |
| Korobov<r_0,r_1> | A polynomial integral transform with weight ∝ x^r_0 * (1-x)^r_1   |
| Korobov<r> | A polynomial integral transform with weight ∝ x^r * (1-x)^r   |
| Sidi<r> | A trigonometric integral transform with weight ∝ sin^r(pi*x) | 
| Baker | The baker's transformation, phi(x) = 1 - abs(2x-1)  |
| None | The trivial transform, phi(x) = x  |

The integral transform used by the qmc can be selected when constructing the qmc.
Example (assuming a real type integrator instance named `integrator`):
```cpp
integrators::Qmc<double,double,10,integrators::transforms::Korobov<5,3>::type> integrator
```
instantiates an integrator which applies a weight `(5,3)` Korobov transform to the integrand before integration.

### Fit Functions

| Name | Description |
| --- | --- |
| PolySingular | A 3rd order polynomial with two additional `1/(p-x)` terms, `f(x) = p_2*(x*(p_0-1))/(p_0-x) + p_3*(x*(p_1-1))/(p_1-x)  + x*(p_4+x*(p_5+x*(1-p_2-p_3-p_4-p_5)))` |
| PolySingularOneSided | A 3rd order polynomial with one additional `1/(p-x)` terms, `f(x) = x*(p_1+x*(p_2+x*p_3)) + (1-p_1-p_2-p_3)*(x*(p_0-1))/(p_0-x)` |

The fit function used by the qmc can be selected when constructing the qmc. These functions are used to approximate the inverse cumulative distribution function of the integrand dimension-by-dimension.
Example (assuming a real type integrator instance named `integrator`):
```cpp
integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> integrator
```
instantiates an integrator which reduces the variance of the integrand by fitting a `PolySingular` type function before integration.

## FAQ

**How do I write the text output of the library to a file?**

First `#include <fstream>`, create a `std::ofstream` instance pointing to your file then set the `logger` of the integrator to the `std::ofstream`. For example to output very detailed output to the file `myoutput.log`:
```cpp
std::ofstream out_file("myoutput.log");
integrators::Qmc<double,double> integrator;
integrator.verbosity=3;
integrator.logger = out_file;
```

**How do I set the seed of the random numbers?**

Set `randomgenerator` to a pseudo-random number engine with the seed you want.
For total reproducability you probably also want to set `cputhreads = 1`  and `devices = {-1}` which disables multi-threading, this helps to ensure that the floating point operations are done in the same order each time the code is run.
For example:
```cpp
integrators::Qmc<double,double,10> integrator;
integrator.randomgenerator.seed(1) // seed = 1
integrator.cputhreads = 1; // no multi-threading
integrator.devices = {-1}; // cpu only
```

**Does this code support multi-threading?**

Yes, if your compiler supports the C++11 thread library then by default the code will try to determine the number of cores or hyper-threads that your hardware supports (via a call to `std::thread::hardware_concurrency()`) and launch this many threads. 
The precise number of threads that will be launched is equal to the `cputhreads` variable.

**I want to integrate another floating point type (e.g. quadruple precision, arbitrary precision, microsoft binary format, etc) can I do that with your code?**

Possibly. Try including the correct header for the type you want and create an instance of the qmc setting the template arguments `T` and `D` to your type.
The following standard library functions must be compatible with your type or a compatible overload must be provided:
* `sqrt`, `abs`, `modf`, `pow`
* `std::max`, `std::min`

If you wish to use certain integal transforms, such as `Korobov` or `Sidi` then your type will also have to support being assigned to a `constexpr` (which is used internally to generate the transforms).
If your type is not intended to represent a real or complex type number then you may also need to overload functions required for calculating the error resulting from the numerical integration, see the files `src/overloads/real.hpp` and `src/overloads/complex.hpp`. Example `9_boost_minimal_demo` demonstrates how to instantiate the qmc with a non-standard type (`boost::multiprecision::cpp_bin_float_quad`), to compile this example you will need the `boost` library available on your system.

**I do not like your generating vectors and/or 100 dimensions and/or 15173222401 lattice points is not enough for me, can I still use your code?**

Yes, but you need to supply your own generating vectors. Compute them using another tool then put them into a map and set `generatingvectors`. For example
```cpp
std::map<unsigned long long int,std::vector<unsigned long long int>> my_generating_vectors = { {7, {1,3}}, {11, {1,7}} };
integrators::Qmc<double,double,10> integrator;
integrator.generatingvectors = my_generating_vectors;
```
If you think your generating vectors will be widely useful for other people then please let us know! With your permission we may include them in the code by default.

**Do you apply an integral transform to make my integrand periodic on [0,1], can I use another one?**

By default we use the polynomial transform of Korobov, `\int_{[0,1]^d} f(\vec{x}) \mathrm{d} \vec{x} = \int_{[0,1]^d} F(\vec{t}) \mathrm{d} \vec{t}` with `F(\vec{t}) := f(\psi(\vec{t})) w_d(\vec{t})` where `w_d(\vec{t}) = \Prod_{j=1}^d w(t_j)`. Specifically we use the `r=3` transform which sets `w(t)=140 t^3 (1-t)^3` and `\psi(t) = 35t^4 -84t^5+70t^6-20t^7` for each variable. If you prefer another integral transform then you can instantiate the integration with another transform. For example:
```cpp
integrators::Qmc<double,double,3,integrators::transforms::Baker::type> integrator;
integrators::result<double> result = integrator.integrate(my_functor);
```

See also the existing transforms in `src/transforms`.

**How does the performance of this code compare to other integration libraries (e.g. CUBA, NIntegrate)?**

TODO - see our paper

**Can I call your code in a similar manner to CUBA routines? Can I call your code from FORTRAN and Mathematica?**

TODO

**Can I call your code from python?**

TODO

## Authors

* Sophia Borowka (@sborowka)
* Gudrun Heinrich (@gudrunhe)
* Stephan Jahn (@jPhy)
* Stephen Jones (@spj101)
* Matthias Kerner (@KernerM)
* Johannes Schlenk (@j-schlenk)
