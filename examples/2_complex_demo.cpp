/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 2_complex_demo.cpp -o 2_complex_demo.out
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 2_complex_demo.cpp -o 2_complex_demo.out
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
#ifdef __CUDACC__
    __host__ __device__
#endif
    complex_t operator()(double* x) const
    {
        return complex_t(1.,1.)*x[0]*x[1]*x[2];
    }
} my_functor;

int main() {
    
    integrators::Qmc<complex_t,double> integrator;
    integrators::result<complex_t> result = integrator.integrate(my_functor,3);
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
