# qmc


A Quasi-Monte-Carlo integrator library.

The library can be used to integrate multi-dimensional real or complex functions numerically.

## Installation

Installation follows the usual `autotools` procedure. 

### With `secdec`

First install mppmu/secdec and set `$SECDEC_CONTRIB` as instructed by the installer.
The Qmc can then be installed into your `$SECDEC_CONTRIB` directory.

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

// TODO

generatingVectors

By default the library uses generating vectors with 100 components, thus it supports integration of functions with up to 100 dimensions.
The implemented Qmc algorithm requires that the generating vectors be generated with a prime lattice size.
The default generating vectors have been generated with lattice size chosen as the next prime number below `2^p` with `p` the natural numbers between 10 to 32. 

// TODO include all generatingVectors in code (currently we do not supply up to 2^32)


## Authors

* Stephan Jahn (@jPhy)
* Stephen Jones (@spj101)
* Matthias Kerner (@KernerM)
