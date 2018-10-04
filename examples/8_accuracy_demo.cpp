/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 8_accuracy_demo.cpp -o 8_accuracy_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 8_accuracy_demo.cpp -o 8_accuracy_demo.out -lgsl -lgslcblas
 */

#include <iostream>
#include <limits> // numeric_limits
#include <cmath> // sin, cos, exp

#include "../src/qmc.hpp"

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

const unsigned int MAXVAR = 5;

template< typename I1, typename I2, typename F>
int count_fails(I1& integrator_fit, I2& integrator_nofit, F& function, unsigned long long int iterations, double true_result)
{
    unsigned fail = 0;
    integrators::result<double> result;
    integrators::fitfunctions::PolySingularTransform<F,double,MAXVAR> fitted_function = integrator_fit.fit(function);
    for(int i = 0; i<iterations; i++)
    {
        result = integrator_nofit.integrate(fitted_function);
//        std::cout << (true_result-result.integral) << " " << result.error << std::endl;
        if ( std::abs( (true_result-result.integral) ) > std::abs(result.error)  )
        {
            fail++;
        }
    }
    return fail;
};

int main() {

    std::cout << "numeric_limits<double>::epsilon() " << std::numeric_limits<double>::epsilon() << std::endl;

    // Integrands
    struct { const int number_of_integration_variables = 1; HOSTDEVICE double operator()(double x[]) const { return x[0]; }; } function1;
    struct { const int number_of_integration_variables = 2; HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]; }; } function2;
    struct {
        const int number_of_integration_variables = 3;
        const double pi = 3.1415926535897932384626433832795028841971693993751;
        HOSTDEVICE double operator()(double x[]) const { return sin(pi*x[0])*cos(pi/2.*x[1])*(1.-cos(pi/4.*x[2])); };
    } function3;
    struct { const int number_of_integration_variables = 4; HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3]); }; } function4;
    struct {
        const int number_of_integration_variables = 5;
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

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> integrator_fit;
    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator_nofit;

    integrator_nofit.evaluateminn = 0;
    integrator_nofit.minn = 8191;

    std::cout << "-- Function 1 --" << std::endl;
    fail1 = count_fails(integrator_fit,integrator_nofit,function1,iterations,function1_result);

    std::cout << "-- Function 2 --" << std::endl;
    fail2 = count_fails(integrator_fit,integrator_nofit,function2,iterations,function2_result);

    std::cout << "-- Function 3 --" << std::endl;
    fail3 = count_fails(integrator_fit,integrator_nofit,function3,iterations,function3_result);

    std::cout << "-- Function 4 --" << std::endl;
    fail4 = count_fails(integrator_fit,integrator_nofit,function4,iterations,function4_result);

    std::cout << "-- Function 5 --" << std::endl;
    fail5 = count_fails(integrator_fit,integrator_nofit,function5,iterations,function5_result);

    std::cout << "-- Summary --" << std::endl;
    std::cout << fail1 << " " << static_cast<double>(fail1)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail2 << " " << static_cast<double>(fail2)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail3 << " " << static_cast<double>(fail3)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail4 << " " << static_cast<double>(fail4)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail5 << " " << static_cast<double>(fail5)/ static_cast<double>(iterations) << std::endl;


};
