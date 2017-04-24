# qmc

A Quasi-Monte-Carlo integrator library.

The library can be used to integrate multi-dimensional real or complex functions numerically.
Multi-threading is supported via the C++11 threading library.
Kahan summation is used to reduce the numerical error due to adding many finite precision floating point numbers.

## Installation

Pre-requisites:
* autotools (tested for automake 1.15, autoconf 2.69)
* a c++11 compatible compiler (tested for Apple LLVM version 8.0.0 clang-800.0.42.1, g++ 6.3.1 20170202)

Installation follows the usual autotools procedure. 

### With secdec

First install mppmu/secdec and set `$SECDEC_CONTRIB` as instructed by the installer.
The QMC can then be installed into your `$SECDEC_CONTRIB` directory.

```shell
autoreconf -i
./configure --prefix=${SECDEC_CONTRIB}/install --libdir=${SECDEC_CONTRIB}/lib --includedir=${SECDEC_CONTRIB}/include --bindir=${SECDEC_CONTRIB}/bin
make
make check
make install
```

### Standalone Mode

```shell
autoreconf -i
./configure
make
make check
make install
```

This will create the usual `lib` and `include` in the default paths. 

## Usage

Example:
```cpp
#include <qmc.hpp>
#include <iostream>

double my_function(double x[])
{
    return x[0]*x[1]*x[2];
}

int main() {
    integrators::Qmc<double,double> integrator;
    integrator.minN = 10000; // (optional) set parameters
    integrators::result<double> result = integrator.integrate(my_function,3);
    std::cout << "integral = " << result.integral << ", error = " << result.error << std::endl;
}
```

Compile and Run:
```shell
$ c++ -std=c++11 -lqmc basic_demo.cpp -o basic_demo
```

Output:
```shell
integral = 0.125, error = 4.36255e-11
```

## API Documentation

The Qmc class has 4 template parameters:
* `T` the return type of the  function to be integrated 
* `D` the argument type of the function to be integrated (assumed to be a floating point type) 
* `U` an unsigned int type (default: `unsigned long long int`)
* `G` a C++11 style pseudo-random number engine (default: `std::mt19937_64`)

### Public fields and Member Functions

`G randomGenerator`

A C++11 style pseudo-random number engine. Default: `std::mt19937_64` seeded with a call to `std::random_device`.

`U getN()`

Returns the lattice size `n` that will be used for integration.

`U minN`

The minimum lattice size that should be used for integration. If a lattice of the requested size is not available then `n` will be the size of the next available lattice with at least `minN` points. Default: `8191`.

`U m`

The number of random shifts of the lattice `m` that should be used to estimate the error of the result. Typically 10 to 50. Default: `32`.

`U blockSize`

Controls the memory used to store the result of integrand evaluations and the number of threads launched.
The samples of the function to be integrated are stored in an array with `blockSize*m` elements. If multi-threading is available then `blockSize` many threads will be launched, each thread will compute `n/blockSize` (or `n/blockSize+1` if `n` is not divisible by `blockSize`) points storing their mean in the threads allocated address in the array. This is repeated for each random shift `m`.

`std::map<U,std::vector<U>> generatingVectors`

A map of available generating vectors which can be used to generate a lattice. By default the library uses generating vectors with 100 components, thus it supports integration of functions with up to 100 dimensions.
The implemented QMC algorithm requires that the generating vectors be generated with a prime lattice size.
The default generating vectors have been generated with lattice size chosen as the next prime number below `2^p` with `p` the natural numbers between 10 to 28.

## FAQ

**I want to set the seed of the random numbers, how do I do that?**

Set `randomGenerator` to a pseudo-random number engine with the seed you want.
Probably you also want to set `blockSize = 1` which disabled multi-threading, this helps to ensure that the floating point operations are done in the same order each time the code is run.
For example:
```cpp
integrators::Qmc<double,double> integrator;
integrator.randomGenerator = std::mt19937_64(1); // seed = 1
integrator.blockSize = 1; // no multi-threading
```

**Does this code support multi-threading?**

Yes, if your compiler supports the C++11 thread library then by default the code will try to determine the number of cores or hyper-threads that your hardware supports (via a call to `std::thread::hardware_concurrency()`) and launch this many threads. 
The precise number of threads that will be launched is equal to the `blockSize` variable.

**I want to integrate another floating point type (e.g. quadruple precision, arbitrary precision, microsoft binary format, etc) can I do that with your code?**

Possibly. By default `libqmc.cpp` (the code used to build the library) instantiates instances of type `template class integrators::Qmc< std::complex<double>, double >` and `template class integrators::Qmc< double , double >`. 
Try adding an instance of the type you have in mind and be sure to include the correct header to support this type. 
If this fails then it is likely that your type is not supported by some standard library function such as `std::sqrt`, `std::abs`, `std::isfinite`, `std::modf`, try defining them. 
If this fails then take a look at `qmc_default.cpp` it defines a relatively small number of simple functions that may need to be reimplemented for your specific type. 
If this also fails then you may need to edit `qmc.cpp`, it is a relatively short code and we hope it is quite easy to understand and modify.

**I do not like your generating vectors and/or 100 dimensions and/or 268435399 lattice points is not enough for me, can I still use your code?**

Yes, but you need to supply your own generating vectors. Compute them using another tool then put them in a map and set `generatingVectors`. For example
```cpp
std::map<unsigned long long int,std::vector<unsigned long long int>> myGeneratingVectors = { {7, {1,3}}, {11, {1,7}} };
integrators::Qmc<double,double> integrator;
integrator.generatingVectors = myGeneratingVectors;
```
If you think your generating vectors will be widely useful for other people then please let us know!
With your permission we may include them in the code by default.

**Do you apply an integral transform to make my integrand periodic on [0,1], can I use another one?**

By default we use the polynomial transform of Korobov, `\int_{[0,1]^d} f(\vec{x}) \mathrm{d} \vec{x} = \int_{[0,1]^d} F(\vec{t}) \mathrm{d} \vec{t}` with `F(\vec{t}) := f(\psi(\vec{t})) w_d(\vec{t})` where `w_d(\vec{t}) = \Prod_{j=1}^d w(t_j)`. Specifically we use the `r=3` transform which sets `w(t)=140 t^3 (1-t)^3` and `\psi(t) = 35t^4 -84t^5+70t^6-20t^7` for each variable. If you prefer another integral transform inherit from the `Qmc` class and implement the protected virtual function `virtual void integralTransform(std::vector<D>& x, D& wgt, const U dim) const` to do what you want.
```cpp
// TODO, also add test case and demo
```

**How does the performance of this code compare to other integration libraries (e.g. CUBA, NIntegrate)?**

TODO

**Can I call your code in a similar manner to CUBA routines? Can I call your code from FORTRAN and Mathematica?**

TODO

**Can I call your code from python?**

TODO

## Authors

* Stephan Jahn (@jPhy)
* Stephen Jones (@spj101)
* Matthias Kerner (@KernerM)
