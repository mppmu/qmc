/*
 * Compile without GPU support:
 *   c++ -std=c++11 -O -I../src 9_boost_minimal_demo.cpp -o 9_boost_minimal_demo.out -lboost_system
 * Compile with GPU support:
 *   (not supported as boost::multiprecision::cpp_bin_float_quad has no cuda implementation)
 */

#include <iostream>
#include <boost/math/special_functions/modf.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "qmc.hpp"

typedef boost::multiprecision::cpp_bin_float_quad quad;

struct my_functor_t {
#ifdef __CUDACC__
    __host__ __device__
#endif
    quad operator()(quad* x) const
    {
        return x[0]*x[1]*x[2];
    }
} my_functor;

int main() {

    integrators::Qmc<quad,quad,unsigned long long int,boost::random::mt19937,boost::random::uniform_real_distribution<quad>> integrator;
    integrator.minn = 1000000;
    integrator.devices = {-1}; // quad only implemented for cpu
    integrators::result<quad,unsigned long long int> result = integrator.integrate(my_functor, 3);
    std::cout << "integral = " << result.integral << std::endl;
    std::cout << "error    = " << result.error    << std::endl;

    return 0;
}
