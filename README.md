qmc
===

A Quasi-Monte-Carlo integrator library.

The library can be used to integrate multi-dimensional real or complex functions numerically.

Installation
------------

Installation follows the usual `autotools` procedure. 

With `secdec`
^^^^^^^^^^^^^

First install mppmu/secdec and set `$SECDEC_CONTRIB` as instructed by the installer.
The Qmc can then be installed into your `$SECDEC_CONTRIB` directory.

```shell
autoreconf -i
./configure --prefix=$(SECDEC_CONTRIB)
make
make check
make install
```

Standalone Mode
^^^^^^^^^^^^^^^

```shell
autoreconf -i
./configure
make
make check
make install
```

This will create the usual `lib` and `include` folders. 
Put them into the relevant paths for your system or include them when compiling your own code.

Usage
-----

```cpp
#include <qmc.hpp>
int main()
{
// TODO - example
}
```

API Documentation
-----------------

# TODO

generatingVectors

By default the library uses generating vectors with 100 components, thus it supports integration of functions with up to 100 dimensions.
The implemented Qmc algorithm requires that the generating vectors be generated with a prime lattice size.
The default generating vectors have been generated with lattice size chosen as the next prime number below 2^p with p the natural numbers between 10--32.


Authors
-------

* Stephan Jahn
* Stephen Jones
* Matthias Kerner
