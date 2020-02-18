/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 2_complex_demo.cpp -o 2_complex_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 2_complex_demo.cpp -o 2_complex_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include "qmc.hpp"

#ifdef __CUDACC__
#include <thrust/complex.h>
typedef thrust::complex<double> complex_t;
#else
#include <complex>
typedef std::complex<double> complex_t;
#endif

struct my_functor_t {
    const unsigned long long int number_of_integration_variables = 3;
#ifdef __CUDACC__
    __host__ __device__
#endif
    complex_t operator()(double* x) const
    {
        return complex_t(1.,1.)*x[0]*x[1]*x[2];
    }
} my_functor;

int main() {

    const unsigned int MAXVAR = 3;
    
    integrators::Qmc<complex_t,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
    integrators::result<complex_t> result = integrator.integrate(my_functor);
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
