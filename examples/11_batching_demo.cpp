/*
 * Compile without GPU support:
 *   c++ -std=c++17 -pthread -I../src 11_batching_demo.cpp -o 11_batching_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++17 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 11_batching_demo.cpp -o 11_batching_demo.out -lgsl -lgslcblas
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
#ifdef __CUDACC__
    __host__
#endif
    void operator()(double* x, double* res, unsigned long long int batchsize)
    {
        // Compute batch of points
        for (unsigned long long int i = 0; i != batchsize; ++i) {
            res[i] = operator()(x + i * number_of_integration_variables);
        }
        return;
    }
    void operator()(){};
} my_functor;

int main() {

    const unsigned int MAXVAR = 3;

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
    integrator.batching = true; // enable batching
    integrator.maxnperpackage = 128; // set maximum batch size
    integrators::result<double> result = integrator.integrate(my_functor);    
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
