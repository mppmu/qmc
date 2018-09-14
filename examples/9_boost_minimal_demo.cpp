/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -O3 -I../src 9_boost_minimal_demo.cpp -o 9_boost_minimal_demo.out -lboost_system -lgsl -lgslcblas
 * Compile with GPU support:
 *   (not supported as boost::multiprecision::cpp_bin_float_quad has no cuda implementation)
 */

#include <iostream>
#include <iomanip> // setprecision
#include <boost/math/special_functions/modf.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "qmc.hpp"

typedef boost::multiprecision::cpp_bin_float_quad quad;

struct my_functor_t {
    const unsigned long long int number_of_integration_variables = 2;
#ifdef __CUDACC__
    __host__ __device__
#endif
    quad operator()(quad* x) const
    {
        return sin(x[0]*x[1])/log(quad(1)+x[0]+x[1]);
    }
} my_functor;

int main() {

    using G = boost::random::mt19937;;
    using H = boost::random::uniform_real_distribution<quad>;
    
    const unsigned int MAXVAR = 2;

    integrators::Qmc<quad,quad,MAXVAR,integrators::transforms::Korobov<10>::type,integrators::fitfunctions::None::type,G,H> integrator;

    integrator.minn = 10000;
    integrator.devices = {-1}; // quad only implemented for cpu
    integrators::result<quad> result = integrator.integrate(my_functor);
    std::cout << std::setprecision(36) << std::endl;
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
