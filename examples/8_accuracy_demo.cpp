/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 8_accuracy_demo.cpp -o 8_accuracy_demo.out
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 8_accuracy_demo.cpp -o 8_accuracy_demo.out
 */



// TODO - fix this example

#include <iostream>
#include <limits> // numeric_limits
#include <cmath> // sin, cos, exp

#include "../src/qmc.hpp"

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

template< typename I, typename F>
int count_fails(I& integrator, F& function, unsigned long long int dimension, unsigned long long int iterations, double true_result)
{
    unsigned fail = 0;
    integrators::result<double,unsigned long long> result;
    for(int i = 0; i<iterations; i++)
    {
        result = integrator.integrate(function,dimension);
//        std::cout << (true_result-result.integral) << " " << result.error;
        if ( std::abs( (true_result-result.integral) ) < std::abs(result.error)  )
        {
//            std::cout << " OK" << std::endl;
        } else
        {
            fail++;
//            std::cout << " FAIL" << std::endl;
        }
    }
    return fail;
};

int main() {

    std::cout << "numeric_limits<double>::epsilon() " << std::numeric_limits<double>::epsilon() << std::endl;

    // Integrands
    struct { const int dim = 1; HOSTDEVICE double operator()(double x[]) const { return x[0]; }; } function1;
    struct { const int dim = 2; HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]; }; } function2;
    struct {
        const int dim = 3;
        const double pi = 3.1415926535897932384626433832795028841971693993751;
        HOSTDEVICE double operator()(double x[]) const { return sin(pi*x[0])*cos(pi/2.*x[1])*(1.-cos(pi/4.*x[2])); };
    } function3;
    struct { const int dim = 4; HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3]); }; } function4;
    struct {
        const int dim = 5;
        const double pi = 3.1415926535897932384626433832795028841971693993751;
        HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3])*cos(pi/2*x[4]); };
    } function5;
    
    // Analytic results
    const double pi = 3.1415926535897932384626433832795028841971693993751;
    const double function1_result = 1./2.;
    const double function2_result = 1./4.;
    const double function3_result = 4./(pi*pi*pi)*(-2.*sqrt(2)+pi);
    const double function4_result = 1./6.*log(3./2.);
    const double function5_result = 1./(6.*pi)*log(9./4.);

    unsigned long long int iterations = 1000;
    unsigned long long int fail1;
    unsigned long long int fail2;
    unsigned long long int fail3;
    unsigned long long int fail4;
    unsigned long long int fail5;
    
    integrators::Qmc<double,double> integrator;
    integrator.minn = 8191;
    integrator.cputhreads = 1;

    std::cout << "-- Function 1 --" << std::endl;
    fail1 = count_fails(integrator,function1,function1.dim,iterations,function1_result);

    std::cout << "-- Function 2 --" << std::endl;
    fail2 = count_fails(integrator,function2,function2.dim,iterations,function2_result);

    std::cout << "-- Function 3 --" << std::endl;
    fail3 = count_fails(integrator,function3,function3.dim,iterations,function3_result);

    std::cout << "-- Function 4 --" << std::endl;
    fail4 = count_fails(integrator,function4,function4.dim,iterations,function4_result);

    std::cout << "-- Function 5 --" << std::endl;
    fail5 = count_fails(integrator,function5,function5.dim,iterations,function5_result);

    std::cout << "-- Summary --" << std::endl;
    std::cout << fail1 << " " << static_cast<double>(fail1)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail2 << " " << static_cast<double>(fail2)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail3 << " " << static_cast<double>(fail3)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail4 << " " << static_cast<double>(fail4)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail5 << " " << static_cast<double>(fail5)/ static_cast<double>(iterations) << std::endl;


};
