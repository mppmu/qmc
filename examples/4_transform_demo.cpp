/*
 * Compile without GPU support:
 *   c++ -std=c++11 -I../src 4_transform_demo.cpp -o 4_transform_demo.out
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 4_transform_demo.cpp -o 4_transform_demo.out
 */

#include <iostream>
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

    integrators::transforms::Tent<double,unsigned long long int> transform;

    integrators::Qmc<double,double> integrator;
    integrators::result<double> result = integrator.integrate(my_functor, 3, transform);

    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
