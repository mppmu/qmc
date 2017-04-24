#include <qmc.hpp>
#include <iostream>

double my_function(double x[])
{
    return x[0]*x[1]*x[2];
}

int main() {
    integrators::Qmc<double,double> integrator;
    integrator.minN = 10000; // (optional) set parameters
    integrators::result<double> result = integrator.integrate(my_function,3);
    std::cout << "integral = " << result.integral << ", error = " << result.error << std::endl;
}
