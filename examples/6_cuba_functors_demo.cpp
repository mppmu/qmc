/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 6_cuba_functors_demo.cpp -o 6_cuba_functors_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 6_cuba_functors_demo.cpp -o 6_cuba_functors_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <cmath> // sin, cos, exp
#include "qmc.hpp"

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif
struct functor1_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return sin(x[0])*cos(x[1])*exp(x[2]); } } functor1;
struct functor2_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return 1./( (x[0] + x[1])*(x[0] + x[1]) + .003 )*cos(x[1])*exp(x[2]); } } functor2;
struct functor3_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return 1./(3.75 - cos(M_PI*x[0]) - cos(M_PI*x[1]) - cos(M_PI*x[2])); } } functor3;
struct functor4_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return fabs(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - .125); } } functor4;
struct functor5_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return exp(-x[0]*x[0] - x[1]*x[1] - x[2]*x[2]); } } functor5;
struct functor6_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return 1./(1. - x[0]*x[1]*x[2] + 1e-10); } } functor6;
struct functor7_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return sqrt(fabs(x[0] - x[1] - x[2])); } } functor7;
struct functor8_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return exp(-x[0]*x[1]*x[2]); } } functor8;
struct functor9_t  { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return x[0]*x[0]/(cos(x[0] + x[1] + x[2] + 1.) + 5.); } } functor9;
struct functor10_t { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return ( (x[0] > .5) ? 1./sqrt(x[0]*x[1]*x[2] + 1e-5) : sqrt(x[0]*x[1]*x[2]) ); } } functor10;
struct functor11_t { const unsigned long long int number_of_integration_variables = 3; HOSTDEVICE double operator()(double* x) const { return ( ((x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) < 1.) ? 1. : 0. ); } } functor11;

template<typename Q, typename I>
void integrate_and_print(Q& real_integrator, I& functor)
{
    integrators::result<double> real_result = real_integrator.integrate(functor);
    std::cout << real_result.integral << " " << real_result.error << std::endl;
}

int main() {

    const unsigned int MAXVAR = 3;

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> real_integrator;

    real_integrator.minn = 10000;
    real_integrator.maxeval = 1;

    integrate_and_print(real_integrator, functor1);
    integrate_and_print(real_integrator, functor2);
    integrate_and_print(real_integrator, functor3);
    integrate_and_print(real_integrator, functor4);
    integrate_and_print(real_integrator, functor5);
    integrate_and_print(real_integrator, functor6);
    integrate_and_print(real_integrator, functor7);
    integrate_and_print(real_integrator, functor8);
    integrate_and_print(real_integrator, functor9);
    integrate_and_print(real_integrator, functor10);
    integrate_and_print(real_integrator, functor11);

}
