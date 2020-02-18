/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 3_defaults_demo.cpp -o 3_defaults_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 3_defaults_demo.cpp -o 3_defaults_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <limits> // numeric_limits
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
    using D = double;
    using U = unsigned long long int;

    const unsigned int MAXVAR = 3;

    integrators::Qmc<D,D,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;

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
    integrator.generatingvectors = integrators::generatingvectors::cbcpt_dn1_100();
    integrator.verbosity = 0;
    integrator.evaluateminn = 100000;
    integrator.fitstepsize = 10;
    integrator.fitmaxiter = 40;
    integrator.fitxtol = 3e-3;
    integrator.fitgtol = 1e-4;
    integrator.fitftol = 1e-8;
//    integrator.fitparametersgsl = {}; // Default constructed

    integrators::result<D> result = integrator.integrate(my_functor);

    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
