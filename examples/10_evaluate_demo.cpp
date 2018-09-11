/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 10_evaluate_demo.cpp -o 10_evaluate_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 10_evaluate_demo.cpp -o 10_evaluate_demo.out -lgsl -lgslcblas
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
    using U = unsigned long long int; // unsinged integer type
    using D = double; // input base type of the integrand
    using T = double; // return type of the integrand

    integrators::Qmc<D,D> integrator;

    // Relevant settings set to their default value
    integrator.logger = std::cout;
    integrator.randomgenerator = std::mt19937_64();
    integrator.minnevaluate = 100000;
    integrator.cputhreads = std::thread::hardware_concurrency();
    integrator.cudablocks = 1024;
    integrator.cudathreadsperblock = 256;
    integrator.devices = {-1}; // devices = cpu (Note: default is actually all devices {-1,0,1,...} detected on construction)
    integrator.generatingvectors = integrators::generatingvectors::cbcpt_dn1_100<U>();
    integrator.verbosity = 0;

    integrators::samples<D,D> samples = integrator.evaluate(my_functor, 3);

    std::cout << "generating vector samples.z = " << std::endl;
    for (const U& item : samples.z)
        std::cout << item << ", ";
    std::cout << std::endl;

    std::cout << "random shift samples.d = " << std::endl;
    for (const D& item : samples.d)
        std::cout << item << ", ";
    std::cout << std::endl;

    std::cout << "samples samples.r = " << std::endl;
    for (const auto& item : samples.r)
        std::cout << item << ", ";
    std::cout << std::endl;

    std::cout << "samples.n = " << samples.n << std::endl;

    return 0;
}
