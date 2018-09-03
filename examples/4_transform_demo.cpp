/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 4_transform_demo.cpp -o 4_transform_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 4_transform_demo.cpp -o 4_transform_demo.out -lgsl -lgslcblas
 */

#include <iostream>
#include "qmc.hpp"

struct my_functor_t {
    const unsigned long long int dim = 3;
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double* x) const
    {
        return x[0]*x[1]*x[2];
    }
} my_functor;

int main() {

    integrators::transforms::Baker<my_functor_t,double,unsigned long long int> transformed_functor(my_functor);

    integrators::Qmc<double,double> integrator;
    integrators::result<double> result = integrator.integrate(transformed_functor);

    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
