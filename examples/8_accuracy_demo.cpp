/*
 * Compile without GPU support:
 *   c++ -std=c++11 -O3 -pthread -I../src 8_accuracy_demo.cpp -o 8_accuracy_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -O3 -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 8_accuracy_demo.cpp -o 8_accuracy_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <limits> // numeric_limits
#include <cmath> // sin, cos, exp
#include <string>

#include "../src/qmc.hpp"

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

// Integrands
const unsigned int MAXVAR = 5;
struct function1_t { const int number_of_integration_variables = 1; HOSTDEVICE double operator()(double x[]) const { return x[0]; }; } function1;
struct function2_t { const int number_of_integration_variables = 2; HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]; }; } function2;
struct function3_t {
    const int number_of_integration_variables = 3;
    const double pi = 3.1415926535897932384626433832795028841971693993751;
    HOSTDEVICE double operator()(double x[]) const { return sin(pi*x[0])*cos(pi/2.*x[1])*(1.-cos(pi/4.*x[2])); };
} function3;
struct function4_t { const int number_of_integration_variables = 4; HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3]); }; } function4;
struct function5_t {
    const int number_of_integration_variables = 5;
    const double pi = 3.1415926535897932384626433832795028841971693993751;
    HOSTDEVICE double operator()(double x[]) const { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3])*cos(pi/2*x[4]); };
} function5;

template< typename I1, typename I2, typename F>
void count_fails(I1& integrator_fit, I2& integrator_nofit, F& function, unsigned long long int iterations, double true_result, int& fail, int& machine_precision)
{
    integrators::result<double> result;
    integrators::fitfunctions::PolySingularTransform<F,double,MAXVAR> fitted_function = integrator_fit.fit(function);
    for(int i = 0; i<iterations; i++)
    {
        result = integrator_nofit.integrate(fitted_function);
        // Check if true result is within error estimate
        if ( std::abs( (true_result-result.integral) ) > std::abs(result.error)  )
        {
            fail++;
        }
        // Check if result reaches machine precision
        if ( std::abs( (true_result-result.integral) ) < 2. * std::numeric_limits<double>::epsilon()  )
        {
            machine_precision++;
        }
    }
};

void print_table_line(std::string name, int iterations, int success, double success_percent, int machine_precision)
{
    const char separator = ' ';
    const int width      = 20;
    std::cout << std::left << std::setw(width) << std::setfill(separator) << name;
    std::cout << std::left << std::setw(width) << std::setfill(separator) << std::to_string(success) + "/" + std::to_string(iterations);
    std::cout << std::left << std::setw(width) << std::setfill(separator) << success_percent;
    std::cout << std::left << std::setw(width) << std::setfill(separator) << std::to_string(machine_precision) + "/" + std::to_string(iterations) << std::endl;
}

int main() {

    std::cout << "numeric_limits<double>::epsilon() " << std::numeric_limits<double>::epsilon() << std::endl;

    // Analytic results
    const double pi = 3.1415926535897932384626433832795028841971693993751;
    const double function1_result = 1./2.;
    const double function2_result = 1./4.;
    const double function3_result = 4./(pi*pi*pi)*(-2.*sqrt(2)+pi);
    const double function4_result = 1./6.*log(3./2.);
    const double function5_result = 1./(6.*pi)*log(9./4.);

    int iterations = 1000;
    int fail1 = 0, fail2 = 0, fail3 = 0, fail4 = 0, fail5 = 0;
    int machine_precision1 = 0, machine_precision2 = 0, machine_precision3 = 0, machine_precision4 = 0, machine_precision5 = 0;

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> integrator_fit;
    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator_nofit;

    integrator_nofit.minn = 8191;
    integrator_nofit.maxeval = 1;

    std::cout << "- Integrating Function 1" << std::endl;
    count_fails(integrator_fit,integrator_nofit,function1,iterations,function1_result,fail1,machine_precision1);

    std::cout << "- Integrating Function 2" << std::endl;
    count_fails(integrator_fit,integrator_nofit,function2,iterations,function2_result,fail2,machine_precision2);

    std::cout << "- Integrating Function 3" << std::endl;
    count_fails(integrator_fit,integrator_nofit,function3,iterations,function3_result,fail3,machine_precision3);

    std::cout << "- Integrating Function 4" << std::endl;
    count_fails(integrator_fit,integrator_nofit,function4,iterations,function4_result,fail4,machine_precision4);

    std::cout << "- Integrating Function 5" << std::endl;
    count_fails(integrator_fit,integrator_nofit,function5,iterations,function5_result,fail5,machine_precision5);

    std::cout << "- Summary" << std::endl;
    const char separator = ' ';
    const int width      = 20;
    std::cout << std::left << std::setw(width) << std::setfill(separator) << "# Name";
    std::cout << std::left << std::setw(width) << std::setfill(separator) << "Success";
    std::cout << std::left << std::setw(width) << std::setfill(separator) << "Success (%)";
    std::cout << std::left << std::setw(width) << std::setfill(separator) << "Machine Precision" << std::endl;

    print_table_line("Function 1",iterations,iterations-fail1, 100. * static_cast<double>(iterations-fail1)/ static_cast<double>(iterations), machine_precision1);
    print_table_line("Function 2",iterations,iterations-fail2, 100. * static_cast<double>(iterations-fail2)/ static_cast<double>(iterations), machine_precision2);
    print_table_line("Function 3",iterations,iterations-fail3, 100. * static_cast<double>(iterations-fail4)/ static_cast<double>(iterations), machine_precision3);
    print_table_line("Function 4",iterations,iterations-fail4, 100. * static_cast<double>(iterations-fail4)/ static_cast<double>(iterations), machine_precision4);
    print_table_line("Function 5",iterations,iterations-fail5, 100. * static_cast<double>(iterations-fail5)/ static_cast<double>(iterations), machine_precision5);

};
