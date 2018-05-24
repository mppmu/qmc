



#define QMC_DEBUG 2



#include "../src/qmc.hpp"
#include <iostream>

struct my_functor {
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double* x) const
    {
        return x[0]*x[1]*x[2];
    }
};

int main() {
    
    integrators::Qmc<double,double> integrator;
    integrator.minn = 10000; // (optional) set parameters
    my_functor my_functor_instance;
    integrators::result<double> result = integrator.integrate(my_functor_instance,3);
    std::cout << "integral = " << result.integral << ", error = " << result.error << std::endl;
}
