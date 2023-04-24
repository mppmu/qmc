/*
 * Compile without GPU support:
 *   c++ -std=c++17 -pthread -I../src 12_generatingvectors_medianQmc_demo.cpp -o 12_generatingvectors_medianQmc_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++17 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 12_generatingvectors_medianQmc_demo.cpp -o 12_generatingvectors_medianQmc_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
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

    const unsigned int MAXVAR = 3;

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
    integrator.generatingvectors = integrators::generatingvectors::none();
    integrator.keepMedianGV = true; // keep newly generating vector for subsequent integrations, only recommended if integrands are similar
    integrator.minn = 100;
    integrator.epsrel = 1e-6;


    integrators::result<double> result = integrator.integrate(my_functor);
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;
    std::cout << "generating vectors:" << std::endl;
    for (auto &gv : integrator.generatingvectors) {
        std::cout << gv.first << ":"; for (auto i : gv.second) std::cout << " " << i; std::cout << std::endl; 
    }
    std::cout << std::endl;


    // repeat integration; no new generating vectors are constructed if integrator.keepMedianGV is set
    result = integrator.integrate(my_functor);
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;
    std::cout << "generating vectors:" << std::endl;
    for (auto &gv : integrator.generatingvectors) {
        std::cout << gv.first << ":"; for (auto i : gv.second) std::cout << " " << i; std::cout << std::endl; 
    }



    return 0;
}
