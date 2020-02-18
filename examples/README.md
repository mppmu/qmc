# Examples

## Compiling

Compile without GPU support:
```shell
c++ -std=c++11 -pthread -I../src <example_name>.cpp -o <example_name>.out
```

Compile with GPU support:
```shell
nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src <example_name>.cpp -o <example_name>.out -lgsl -lgslcblas
```
Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).

## Introductory Examples

The following examples demonstrate the basic usage of the integrator.

### 1_minimal_demo

Demonstrates the simplest usage of the QMC. Shows how to integrate the 3 dimensional function `f(x0,x1,x2) = x0*x1*x2`. The CUDA function execution space specifiers `__host__` and  `__device__` are present only when compiling with GPU support, this is controlled by the presence of the `__CUDACC__` macro which is automatically defined by the compiler during CUDA compilation.

### 2_complex_demo

Demonstrates how to integrate a complex function in a way compatible with both CPU and GPU evaluation. Since the type `std::complex` is not currently supported by GPUs we use the type `thrust::complex` if compiling with GPU support, this is achieved using a `typedef` which depends on the presence of `__CUDACC__`.

### 3_defaults_demo

Demonstrates how to adjust all settings of the QMC. In this demo all values are set to their default.

### 4_fitfunctions_demo

Demonstrates how to change the fit function used by the integator. The available fit functionsare listed in `src/fitfunctions`. In this example select the `PolySingular` fitfunction when instantiating the integrator, this overrides the default fit function and instead causes the integrator to attempt to reduce the variance of the input function using the `PolySingular` ansatz.

### 5_generatingvectors_demo

Demonstrates how to change the generating vectors used by the integrator. The available generating vectors are listed in `src/generatingvectors`. In this example we switch the generating vectors used by the integrators from the default to `cbcpt_dn2_6`.

### 6_cuba_functors_demo

A translation of the CUBA demo file for input to the QMC. For each function in the CUBA test suite we create a corresponding functor. In the `main` function we instantiate a copy of the QMC capable of integrating real functions. Finally we pass each functor to the integrator and print the result using the `integrate_and_print` function.

### 7_cuba_pointers_demo

An alternative translation of the CUBA demo file for input to the QMC. In  `6_cuba_functors_demo` a new functor is declared for each function in the CUBA test suite, in particular, each functor has a different type which corresponds to the function it contains. However, in `c++11` it can sometimes be inconvenient to iterate over objects of different types. In this demo we show how a functor of a single type can be used to refer to each of the functions in the test suite. This is a 4 stage process:
1. Declare the functions to be integrated, e.g.
```cpp
__host__ __device__ func1(double x[]) { return sin(x[0])*cos(x[1])*exp(x[2]); }
```
2. Create pointers to these functions on each of the CUDA devices, as of CUDA 9.0 this must be done in global scope, e.g.
```cpp
__device__ integrand* device_func1 = func1; // must be in global scope
```
3. Create functions which can copy the device function pointers to the host, as of CUDA 9.0 this must be done in global scope due to the call to `cudaMemcpyFromSymbol`, e.g.
```cpp
integrand* get_device_address_func1()
{
#ifdef __CUDACC__
integrand* device_address_on_host;
CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func1, sizeof(integrand*))); // must be called by a function in global scope
return device_address_on_host;
#else
return nullptr;
#endif
}
```
4. Create the functor that will hold these function pointers with a call operator to evaluate the function pointed to:
```cpp
struct my_functor_t {
get_device_address* device_address_getter;

integrand* host_address;
integrand* device_address;

#ifdef __CUDACC__
__host__ __device__
#endif
double operator()(double* x) const
{
#ifdef __CUDA_ARCH__
return device_address(x);
#else
return host_address(x);
#endif
}

//Copy constructor
my_functor_t(const my_functor_t& my_functor):
device_address_getter(my_functor.device_address_getter),
host_address(my_functor.host_address),
device_address(my_functor.device_address_getter()) // update device_address for current device
{};

// Constructor
my_functor_t(integrand* host_address, get_device_address* device_address_getter):
device_address_getter(device_address_getter),
host_address(host_address),
device_address(device_address_getter())
{};

};
```
Note that this functor holds 3 pointers: 
* a pointer to the function which returns the device address (`device_address_getter`)
* a pointer to the host address of the function (`host_address`) and
* a pointer to the function on the current device (`device_address`).

The functor's call operator decides whether to call the host or device version of the function based on the presence of the `__CUDA_ARCH__` preprocessor macro which is automatically defined by the compiler depending on the architecture currently being compiled for (the function will typically be compiled twice by `nvcc`, once for the GPU and once for the CPU). The constructor of the functor takes a host address and a pointer to the device address getter function, the device address is determined via a call to the device address getter. 
 The QMC integrator can change the device on which the functor will be called after the functor is constructed, it is therefore necessary to be able to update the device address to the address on the current device. In the integrator we provide a mechanism to update the device address by always copying the functor before the first call on the device, therefore, a copy constructor can be implemented which updates the device address by making an additional call to `device_address_getter`, as shown above. Finally, in the `main` function we create a vector of functors each pointing to a different function, we iterate over this vector integrating each function and printing the result.
 
 ### 8_accuracy_demo
 
Uses the qmc to integrate 5 functions 1000 times each. After each integration the uncertainty reported by the qmc is compared to the true error (which is computed by comparing the qmc result to the true result). The example prints, for each function, the number of times the true result differs from the result obtained by the qmc by less than the uncertainty reported by the qmc as well as the number of results that were accurate to machine precision. 

Assuming that the uncertainty is Gaussian distributed we would expect the true result to be within one uncertainty of the reported result 68.27% of the time. If the result is accurate to machine precision the result is not Gaussian distributed and often is understimated as the qmc will return (nearly) the same machine precise result for each sample resulting in an incorrectly small variance between the samples.
 
 ### 9_boost_minimal_demo
 
 Demonstrates the use of higher than double precision numbers within the qmc. Here we integrate a `boost` quadruple precision float (`cpp_bin_float_quad`). The `boost` library is required to compile this example.
 
 We integrate the function `sin(x[0]*x[1])/log(quad(1)+x[0]+x[1])` using a weight `r=10` Korobov trasnform. With a lattice size of `~10000` approximately 30 digits are obtained.
  
 ## Loop Integral Examples
 
 The following examples are taken from high energy physics loop integals and demonstrate the usage of the integrator on examples of interest to the authors.
 
 ### 100_hh_sector_demo
 
 A single sector of a sector decomposed 2-loop double higgs integral, taken from [arXiv:1604.06447](https://arxiv.org/abs/1604.06447) and [arXiv:1608.04798](https://arxiv.org/abs/1608.04798). This sector at phase-space point `Run 4160` (hardcoded near the top of the example) was identified by exhaustive search as having particularly poor scaling with integral transform Korobov weight 3 applied.
 
 ### 101_ff4_demo
 
 The finite 4-loop form factor of Eq(7.1) [arXiv:1510.06758](https://arxiv.org/abs/1510.06758). Here, rather than applying sector decomposition (which generates O(20k) sectors) we try to integrate the function directly. The first few lines map the domain of integration from the simplex to the hypercube. The example depends on 11 Feynman parameters (integration variables) and we find that (due to the high dimension?) the variance of the function is extremely large particualrly when the Korobov transforms are applied. Note that although the input function is integrable it is not finite everywhere, in particular it is singular when some of the parameters tend to zero.
 
 The analytic result evalutes to: `3.1807380843134699650...`
 
 ### 102_ff2_demo
 
 A finite 2-loop form factor without sector decompositon, called `INT[“B2diminc4”, 6, 63, 10, 0, {2, 2, 2, 2, 1, 1, 0}]` in [arXiv:1510.06758](https://arxiv.org/abs/1510.06758) (see references therein for the much earlier original calculations). Note that although the input function is integrable it is not finite everywhere, in particular it singular when some of the parameters tend to zero.
 
 The analytic result evalutes to: `0.27621049702196548441...`
 
 ### 103_hj_double_box
 
 A single sector of a sector decomposed 2-loop integral contributing to HJ production, which appeared in an early stage of [arXiv:1802.00349](https://arxiv.org/abs/1802.00349).  More specificly, it is the order epsilon part of sector 18, generated by SecDec 3 with the following settings:

 proplist={(p1)^2-msq, (p1+k1)^2-msq, (p1-k2)^2-msq, (p1-k2-k3)^2-msq, (p2)^2-msq, (p2+k1)^2-msq, (p2-k2)^2-msq, (p2-k2-k3)^2-msq, (p1-p2)^2};
 powerlist={0,1,0,1,1,1,2,0,1}; 
 (k4)^2 = (-k1-k2-k3)^2 = mH^2

 The integral is evaluated at high invariant mass m_{hj} = 8.8 m_t. 
 Due to its bad convergence, a different master integral has been chosen to obtain the final results of the above publication.
 
 ## Profiling Examples
 
 The following examples are used to profile the code.
 
 ### 1000_genz_demo
 
 Computes the test functions of Genz (A. Genz, A package for testing multiple integration subroutines, in: P. Keast, G. Fair-weather (eds.), Numerical Integration, Kluwer, Dordrecht, 1986.) using the qmc as well as the Vegas, Suave, Divonne and Cuhre algorithms as implemented in the [Cuba library](http://www.feynarts.de/cuba/).
 
