/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 101_ff4_demo.cpp -o 101_ff4_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 101_ff4_demo.cpp -o 101_ff4_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <iomanip>

#include "qmc.hpp"

struct formfactor4L_t {
    const unsigned long long int number_of_integration_variables = 11;
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
        double x5  = (1.-x0-x1-x2-x3-x4)*arg[5];
        double x6  = (1.-x0-x1-x2-x3-x4-x5)*arg[6];
        double x7  = (1.-x0-x1-x2-x3-x4-x5-x6)*arg[7];
        double x8  = (1.-x0-x1-x2-x3-x4-x5-x6-x7)*arg[8];
        double x9  = (1.-x0-x1-x2-x3-x4-x5-x6-x7-x8)*arg[9];
        double x10 = (1.-x0-x1-x2-x3-x4-x5-x6-x7-x8-x9)*arg[10];
        double x11 = (1.-x0-x1-x2-x3-x4-x5-x6-x7-x8-x9-x10);

        double wgt =
        (1.-x0)*
        (1.-x0-x1)*
        (1.-x0-x1-x2)*
        (1.-x0-x1-x2-x3)*
        (1.-x0-x1-x2-x3-x4)*
        (1.-x0-x1-x2-x3-x4-x5)*
        (1.-x0-x1-x2-x3-x4-x5-x6)*
        (1.-x0-x1-x2-x3-x4-x5-x6-x7)*
        (1.-x0-x1-x2-x3-x4-x5-x6-x7-x8)*
        (1.-x0-x1-x2-x3-x4-x5-x6-x7-x8-x9);

        if(wgt <= 0.0) return 0;

        double f = x1*(x8*(x3*x4*x6+x4*(x5+x6)*x7+x3*(x4+x5+x6)*x7+(x4+x5+x6)*x7*x9+x11*(x6+x7)*(x3+x4+x9))+x10*((x11*x6+(x11+x5+x6)*x7)*(x8+x9)+x4*x7*(x5+x8+x9)))+x0*(x2*x3*x4*x5+x1*x3*x4*x8+x2*x3*x4*x8+x1*x3*x5*x8+x2*x3*x5*x8+x1*x4*x5*x8+x2*x4*x5*x8+x3*x4*x5*x8+x1*x3*x6*x8+x2*x3*x6*x8+x1*x4*x6*x8+x2*x4*x6*x8+x3*x4*x6*x8+x3*x4*x7*x8+x3*x5*x7*x8+x4*x5*x7*x8+x3*x6*x7*x8+x4*x6*x7*x8+x11*(x3+x4)*(x2*x5+(x1+x2+x5+x6+x7)*x8)+(x4+x5+x6)*(x2*x3+(x1+x2+x3+x7)*x8)*x9+x11*((x1+x3+x4+x5+x6+x7)*x8+x2*(x3+x4+x5+x8))*x9+x10*(x11*(x4*x5+x1*x8+x4*x8+x5*x8+x6*x8+x7*x8+(x1+x4+x5+x6+x7)*x9+x2*(x5+x8+x9)+x3*(x5+x8+x9))+(x1+x2+x3+x7)*((x5+x6)*(x8+x9)+x4*(x5+x8+x9))));
        double u = x10*x11*x2*x5+x10*x11*x3*x5+x11*x2*x3*x5+x10*x11*x4*x5+x10*x2*x4*x5+x11*x2*x4*x5+x10*x3*x4*x5+x2*x3*x4*x5+x10*x11*x2*x6+x10*x11*x3*x6+x11*x2*x3*x6+x10*x11*x4*x6+x10*x2*x4*x6+x11*x2*x4*x6+x10*x3*x4*x6+x2*x3*x4*x6+x10*x11*x2*x7+x10*x11*x3*x7+x11*x2*x3*x7+x10*x11*x4*x7+x10*x2*x4*x7+x11*x2*x4*x7+x10*x3*x4*x7+x2*x3*x4*x7+x10*x2*x5*x7+x10*x3*x5*x7+x2*x3*x5*x7+x10*x4*x5*x7+x2*x4*x5*x7+x10*x2*x6*x7+x10*x3*x6*x7+x2*x3*x6*x7+x10*x4*x6*x7+x2*x4*x6*x7+x10*x11*x2*x8+x10*x11*x3*x8+x11*x2*x3*x8+x10*x11*x4*x8+x10*x2*x4*x8+x11*x2*x4*x8+x10*x3*x4*x8+x2*x3*x4*x8+x10*x11*x5*x8+x10*x2*x5*x8+x10*x3*x5*x8+x11*x3*x5*x8+x2*x3*x5*x8+x11*x4*x5*x8+x2*x4*x5*x8+x3*x4*x5*x8+x10*x11*x6*x8+x10*x2*x6*x8+x10*x3*x6*x8+x11*x3*x6*x8+x2*x3*x6*x8+x11*x4*x6*x8+x2*x4*x6*x8+x3*x4*x6*x8+x10*x11*x7*x8+x11*x3*x7*x8+x10*x4*x7*x8+x11*x4*x7*x8+x3*x4*x7*x8+x10*x5*x7*x8+x3*x5*x7*x8+x4*x5*x7*x8+x10*x6*x7*x8+x3*x6*x7*x8+x4*x6*x7*x8+(x11*x2*(x3+x4+x5+x6+x7)+x10*((x4+x5+x6)*(x2+x3+x7)+x11*(x2+x3+x4+x5+x6+x7))+x11*(x2+x3+x4+x5+x6+x7)*x8+(x4+x5+x6)*(x2*(x3+x7)+(x2+x3+x7)*x8))*x9+x0*(x10*x11*x2+x10*x11*x3+x11*x2*x3+x10*x11*x4+x10*x2*x4+x11*x2*x4+x10*x3*x4+x2*x3*x4+x10*x11*x5+x10*x2*x5+x10*x3*x5+x11*x3*x5+x2*x3*x5+x11*x4*x5+x2*x4*x5+x3*x4*x5+x10*x11*x6+x10*x2*x6+x10*x3*x6+x11*x3*x6+x2*x3*x6+x11*x4*x6+x2*x4*x6+x3*x4*x6+x10*x11*x7+x11*x3*x7+x10*x4*x7+x11*x4*x7+x3*x4*x7+x10*x5*x7+x3*x5*x7+x4*x5*x7+x10*x6*x7+x3*x6*x7+x4*x6*x7+(x4+x5+x6)*(x2+x3+x7)*x9+x11*(x2+x3+x4+x5+x6+x7)*x9+x1*(x3*x4+x3*x5+x4*x5+x3*x6+x4*x6+x10*(x11+x4+x5+x6)+(x4+x5+x6)*x9+x11*(x3+x4+x9)))+x1*(x3*x4*x5+x3*x4*x6+x3*x4*x7+x3*x5*x7+x4*x5*x7+x3*x6*x7+x4*x6*x7+x3*x4*x8+x3*x5*x8+x4*x5*x8+x3*x6*x8+x4*x6*x8+x11*(x3+x4)*(x5+x6+x7+x8)+(x4+x5+x6)*(x3+x7+x8)*x9+x11*(x3+x4+x5+x6+x7+x8)*x9+x10*((x5+x6)*(x7+x8+x9)+x11*(x5+x6+x7+x8+x9)+x4*(x5+x6+x7+x8+x9)));
        double n = x0*x9;
        double d = f*f*u;

        return wgt*n/d;
    }
} formfactor4L;

int main() {

    const unsigned int MAXVAR = 11;

    // fit function to reduce variance
    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> fitter;
    integrators::fitfunctions::PolySingularTransform<formfactor4L_t,double,MAXVAR> fitted_formfactor4L = fitter.fit(formfactor4L);

    // setup integrator
    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
    integrator.minm = 20;
    integrator.maxeval = 1; // do not iterate

    std::cout << "# n m Re[I] Im[I] Re[Abs. Err.] Im[Abs. Err.]" << std::endl;
    std::cout << std::setprecision(16);
    for(const auto& generating_vector : integrator.generatingvectors)
    {
        integrator.minn = generating_vector.first;
        integrators::result<double> result = integrator.integrate(formfactor4L);

        std::cout
        << result.n
        << " " << result.m
        << " " << result.integral
        << " " << result.error
        << std::endl;
    }
}



