/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 10_evaluate_demo.cpp -o 10_evaluate_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 10_evaluate_demo.cpp -o 10_evaluate_demo.out -lgsl -lgslcblas
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
    using U = unsigned long long int; // unsinged integer type
    using D = double; // input base type of the integrand
    using T = double; // return type of the integrand

    const unsigned int MAXVAR = 3;

    integrators::Qmc<D,D,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;

    // Relevant settings set to their default value
    integrator.logger = std::cout;
    integrator.randomgenerator = std::mt19937_64();
    integrator.cputhreads = std::thread::hardware_concurrency();
    integrator.cudablocks = 1024;
    integrator.cudathreadsperblock = 256;
    integrator.devices = {-1}; // devices = cpu (Note: default is actually all devices {-1,0,1,...} detected on construction)
    integrator.generatingvectors = integrators::generatingvectors::cbcpt_dn1_100();
    integrator.verbosity = 0;
    integrator.evaluateminn = 100000;

    integrators::samples<D,D> samples = integrator.evaluate(my_functor);

    std::cout << "generating vector samples.z = " << std::endl;
    for (const U& item : samples.z)
        std::cout << item << ", ";
    std::cout << std::endl;

    std::cout << "random shift samples.d = " << std::endl;
    for (const D& item : samples.d)
        std::cout << item << ", ";
    std::cout << std::endl;

    std::cout << "samples.n = " << samples.n << std::endl;

    const unsigned long long int point_index = 0;
    std::cout << "f(";
    for (int sdim = 0; sdim < my_functor.number_of_integration_variables; sdim++ )
        std::cout << samples.get_x(point_index,sdim) << (sdim == (my_functor.number_of_integration_variables-1) ? "" : ",");
    std::cout << ") = " << samples.r.at(point_index) << std::endl;

    return 0;
}

