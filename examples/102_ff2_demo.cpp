/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 102_ff2_demo.cpp -o 102_ff2_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -std=c++11 -x cu -I../src 102_ff2_demo.cpp -o 102_ff2_demo.out -lgsl -lgslcblas
 */

#include <iostream>
#include <iomanip>
#include <map>

#include "qmc.hpp"

struct formfactor2L_t {
#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(const double arg[]) const
    {

        double x0  = arg[0];
        double x1  = (1.-x0)*arg[1];
        double x2  = (1.-x0-x1)*arg[2];
        double x3  = (1.-x0-x1-x2)*arg[3];
        double x4  = (1.-x0-x1-x2-x3)*arg[4];
        double x5  = (1.-x0-x1-x2-x3-x4);

        double wgt =
        (1.-x0)*
        (1.-x0-x1)*
        (1.-x0-x1-x2)*
        (1.-x0-x1-x2-x3);

        if(wgt <= 0) return 0;

        double u=x2*(x3+x4)+x1*(x2+x3+x4)+(x2+x3+x4)*x5+x0*(x1+x3+x4+x5);
        double f=x1*x2*x4+x0*x2*(x1+x3+x4)+x0*(x2+x3)*x5;
        double n=x0*x1*x2*x3;
        double d = f*f*u*u;

        return wgt*n/d;
    }
} formfactor2L;

int main() {

    using D = double;
    using U = unsigned long long int;

    integrators::Qmc<D,D> integrator;
    integrator.minm = 20;
    integrator.maxeval = 1; // do not iterate

    // Append large generating vectors to default generating vectors
    std::map<U,std::vector<U>> large_vecs = integrators::generatingvectors::cbcpt_cfftw1_6<U>();
    integrator.generatingvectors.insert(large_vecs.begin(),large_vecs.end());

    integrators::transforms::Korobov<double,unsigned long long int,5> integral_transform;

    std::cout << "# n m Re[I] Im[I] Re[Abs. Err.] Im[Abs. Err.]" << std::endl;
    std::cout << std::setprecision(16);

    for(const auto& generating_vector : integrator.generatingvectors)
    {
        integrator.minn = generating_vector.first;
        integrators::result<double> result = integrator.integrate(formfactor2L,5,integral_transform);

        std::cout
        << result.n
        << " " << result.m
        << " " << result.integral
        << " " << result.error
        << std::endl;
    }
}
