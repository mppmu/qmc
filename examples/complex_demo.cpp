#include "../src/qmc.hpp"
#include <iostream>

#ifdef __CUDACC__
#include <thrust/complex.h>
typedef thrust::complex<double> complex_t;
#else
#include <complex>
typedef std::complex<double> complex_t;
#endif

struct my_functor {
#ifdef __CUDACC__
    __host__ __device__
#endif
    complex_t operator()(double* x) const
    {
        return complex_t(1.,1.)*x[0]*x[1]*x[2];
    }
};

//struct my_integral_transform {
//#ifdef __CUDACC__
//__host__ __device__
//#endif
//    void operator()(double* x, double& wgt, const unsigned long long int dim) const
//    {
//    }
//};

int main() {
    
    integrators::Qmc<complex_t,double> integrator;
    integrator.minn = 1000000; // (optional) set parameters
    integrator.epsrel = 1e-9;
    integrator.epsabs = 0;
    integrator.maxeval = 81904*32;
    integrator.verbosity = 3;
    integrator.cputhreads = 4;
    my_functor my_functor_instance;
//    my_integral_transform my_integral_transform_instance;
//    integrators::result<complex_t> result = integrator.integrate(my_functor_instance,3,my_integral_transform_instance);
    integrators::result<complex_t> result = integrator.integrate(my_functor_instance,3);

    std::cout << "integral = " << result.integral << ", error = " << result.error << std::endl;
    
    return 0;
}
