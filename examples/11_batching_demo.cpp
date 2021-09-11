/*
 * Compile without GPU support:
 *   c++ -march=native -O3 -std=c++17 -pthread -I../src 11_batching_demo.cpp -o 11_batching_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++17 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 11_batching_demo.cpp -o 11_batching_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <iomanip> // setprecision
#include <chrono>
#include "qmc.hpp"

struct my_functor_t {
    const unsigned long long int number_of_integration_variables = 2;
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double* x) const
    {
        //return 1.; // Trivial, for measuring overhead of sampling
        // return pow(x[0]+x[1],10.); // Inefficient way to compute powers
        return pow(x[1],10)+ 10*x[0]*pow(x[1],9)+ 45*pow(x[0],2)*pow(x[1],8)+ 120*pow(x[0],3)*pow(x[1],7)+ 210*pow(x[0],4)*pow(x[1],6)+ 252*pow(x[0],5)*pow(x[1],5)+ 210*pow(x[0],6)*pow(x[1],4)+ 120*pow(x[0],7)*pow(x[1],3)+ 45*pow(x[0],8)*pow(x[1],2)+ 10*pow(x[0],9)*x[1]+ pow(x[0],10); // Extremely inefficient way to compute powers
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

    const unsigned int MAXVAR = 2;

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;

    integrator.batching = true; // enable batching
    integrator.maxnperpackage = 8192; // set maximum batch size
    integrator.minn=1000000;
    integrator.cputhreads=1;
    
    integrators::result<double> result;
    std::chrono::steady_clock::time_point time_before;
    std::chrono::steady_clock::time_point time_after;
    double time_in_ns;

    std::cout << std::setprecision(16);
    std::cout << "Integrating without batching" << std::endl;
    integrator.batching = false;
    time_before = std::chrono::steady_clock::now();
    result  = integrator.integrate(my_functor);    
    time_after = std::chrono::steady_clock::now();
    time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_after - time_before).count();
    std::cout << "  integral = " << result.integral << std::endl;
    std::cout << "  error    = " << result.error    << std::endl;
    std::cout << "  time (s) = " << time_in_ns*1e-9  << std::endl;

    std::cout << "Integrating with batching" << std::endl;
    integrator.batching = true;
    time_before = std::chrono::steady_clock::now();
    result  = integrator.integrate(my_functor);
    time_after = std::chrono::steady_clock::now();
    time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_after - time_before).count();
    std::cout << "  integral = " << result.integral << std::endl;
    std::cout << "  error    = " << result.error    << std::endl;
    std::cout << "  time (s) = " << time_in_ns*1e-9  << std::endl;

    return 0;
}
