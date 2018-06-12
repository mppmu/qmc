/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 3_defaults_demo.cpp -o 3_defaults_demo.out
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 3_defaults_demo.cpp -o 3_defaults_demo.out
 */

#include <iostream>
#include <limits> // numeric_limits
#include "qmc.hpp"

struct my_functor_t {
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double* x) const
    {
        return x[0]*x[1]*x[2];
    }
} my_functor;

int main() {
    using D = double;
    using U = unsigned long long int;

    integrators::Qmc<D,D> integrator;

    // All settings set to their default value
    integrator.logger = std::cout;
    integrator.randomgenerator = std::mt19937_64();
    integrator.minn = 8191;
    integrator.minm = 32;
    integrator.epsrel = 0.01;
    integrator.epsabs  = 1e-7;
    integrator.maxeval = 1000000;
    integrator.maxnperpackage = 1;
    integrator.maxmperpackage = 1024;
    integrator.errormode = integrators::ErrorMode::all;
    integrator.cputhreads = std::thread::hardware_concurrency();
    integrator.cudablocks = 1024;
    integrator.cudathreadsperblock = 256;
    integrator.devices = {-1}; // devices = cpu (Note: default is actually all devices {-1,0,1,...} detected on construction)
    integrator.generatingvectors = integrators::generatingvectors::cbcpt_dn1_100<U>();
    integrator.verbosity = 0;

    integrators::result<D> result = integrator.integrate(my_functor, 3);

    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
