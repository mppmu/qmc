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

```cpp
#include <qmc.hpp>
int main()
{
// TODO - example
}
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
