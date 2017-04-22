#include <qmc.hpp>
#include <iostream>
#include <complex>
#include <cmath> // sin, cos, exp

int main() {

    // Cuba demo functions
    std::vector<std::function<double(double[])>> functions =
    {
        [] (double x[]) { return sin(x[0])*cos(x[1])*exp(x[2]); },
        [] (double x[]) { return 1./( (x[0] + x[1])*(x[0] + x[1]) + .003 )*cos(x[1])*exp(x[2]); },
        [] (double x[]) { return 1./(3.75 - cos(M_PI*x[0]) - cos(M_PI*x[1]) - cos(M_PI*x[2])); },
        [] (double x[]) { return fabs(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - .125); },
        [] (double x[]) { return exp(-x[0]*x[0] - x[1]*x[1] - x[2]*x[2]); },
        [] (double x[]) { return 1./(1. - x[0]*x[1]*x[2] + 1e-10); },
        [] (double x[]) { return sqrt(fabs(x[0] - x[1] - x[2])); },
        [] (double x[]) { return exp(-x[0]*x[1]*x[2]); },
        [] (double x[]) { return x[0]*x[0]/(cos(x[0] + x[1] + x[2] + 1.) + 5.); },
        [] (double x[]) { return ( (x[0] > .5) ? 1./sqrt(x[0]*x[1]*x[2] + 1e-5) : sqrt(x[0]*x[1]*x[2]) ); },
        [] (double x[]) { return ( ((x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) < 1.) ? 1. : 0. ); }
    };

    integrators::Qmc<double,double> real_integrator;
    real_integrator.minN = 10000; // TODO - set parameters similar to CUBA demo?
    std::cout << "real_integrator.getN(): " << real_integrator.getN() << std::endl;

    integrators::result<double> real_result;
    for( const std::function<double(double[])>& function : functions )
    {
        real_result = real_integrator.integrate(function,3);
        std::cout << real_result.integral << " " << real_result.error << std::endl;
    }

}
