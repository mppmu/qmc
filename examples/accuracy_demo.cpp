#include <qmc.hpp>
#include <iostream>
#include <complex>
#include <limits> // numeric_limits
#include <cmath> // sin, cos, exp, pow

//// function from http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
//template<class T>
//typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
//almost_equal(T x, T y, int ulp)
//{
//    // the machine epsilon has to be scaled to the magnitude of the values used
//    // and multiplied by the desired precision in ULPs (units in the last place)
//    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
//    // unless the result is subnormal
//    || std::abs(x-y) < std::numeric_limits<T>::min();
//}

int count_fails(integrators::Qmc<double,double> integrator, std::function<double(double[])> function, unsigned dimension, unsigned iterations, double true_result)
{
    unsigned fail = 0;
    integrators::result<double> result;
    for(int i = 0; i<iterations; i++)
    {
        result = integrator.integrate(function,dimension);
        std::cout << (true_result-result.integral) << " " << result.error;
        if ( std::abs( (true_result-result.integral) ) < std::abs(result.error)  )
        {
            std::cout << " OK" << std::endl;
        } else
        {
            fail++;
            std::cout << " FAIL" << std::endl;
        }
        //        else if ( almost_equal( std::abs((true_result-result.integral)), std::abs(result.error), 2 ) )
        //        {
        //            std::cout << " OK (ULP 2)" << std::endl;
        //        }
    }
    return fail;
}

int main() {

    std::cout << std::numeric_limits<double>::epsilon() << std::endl;

    double pi = 3.1415926535897932384626433832795028841971693993751;

    std::function<double(double[])> function1 = [] (double x[]) { return x[0]; };
    std::function<double(double[])> function2 = [] (double x[]) { return x[0]*x[1]; };
    std::function<double(double[])> function3 = [pi] (double x[]) { return std::sin(pi*x[0])*std::cos(pi/2.*x[1])*(1.-std::cos(pi/4.*x[2])); };
    std::function<double(double[])> function4 = [] (double x[]) { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3]); };
    std::function<double(double[])> function5 = [pi] (double x[]) { return x[0]*x[1]*(1.-x[2]*x[2])/(3.-x[3])*std::cos(pi/2*x[4]); };

    // Analytic results
    double function1_result = 1/2.;
    double function2_result = 1./4.;
    double function3_result = 4./std::pow(pi,3)*(-2.*std::sqrt(2)+pi);
    double function4_result = 1./6.*std::log(3./2.);
    double function5_result = 1./(6.*pi)*std::log(9./4.);

    integrators::Qmc<double,double> integrator;
    integrator.minN = 8191;

    unsigned iterations = 1000;
    unsigned fail1;
    unsigned fail2;
    unsigned fail3;
    unsigned fail4;
    unsigned fail5;

    std::cout << "-- Function 1 --" << std::endl;
    fail1 = count_fails(integrator,function1,1,iterations,function1_result);

    std::cout << "-- Function 2 --" << std::endl;
    fail2 = count_fails(integrator,function2,2,iterations,function2_result);

    std::cout << "-- Function 3 --" << std::endl;
    fail3 = count_fails(integrator,function3,3,iterations,function3_result);

    std::cout << "-- Function 4 --" << std::endl;
    fail4 = count_fails(integrator,function4,4,iterations,function4_result);

    std::cout << "-- Function 5 --" << std::endl;
    fail5 = count_fails(integrator,function5,5,iterations,function5_result);

    std::cout << "-- Summary --" << std::endl;
    std::cout << fail1 << " " << static_cast<double>(fail1)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail2 << " " << static_cast<double>(fail2)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail3 << " " << static_cast<double>(fail3)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail4 << " " << static_cast<double>(fail4)/ static_cast<double>(iterations) << std::endl;
    std::cout << fail5 << " " << static_cast<double>(fail5)/ static_cast<double>(iterations) << std::endl;


}
