/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 4_fitfunctions_demo.cpp -o 4_fitfunctions_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 4_fitfunctions_demo.cpp -o 4_fitfunctions_demo.out -lgsl -lgslcblas
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
        return x[0]*x[1]*x[2]; // TODO - function where fit helps?
    }
} my_functor;

int main() {

    const unsigned int MAXVAR = 3;

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> integrator;
    integrators::result<double> result = integrator.integrate(my_functor);

    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
