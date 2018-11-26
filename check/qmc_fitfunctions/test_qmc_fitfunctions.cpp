#include "catch.hpp"
#include "qmc.hpp"

#include<iostream>

#ifdef __CUDACC__
#include <thrust/complex.h>
using thrust::complex;
#define HOSTDEVICE __host__ __device__
#else
using std::complex;
#include <complex>
#define HOSTDEVICE
#endif

struct simple_function_t {
    const unsigned long long int number_of_integration_variables = 2;
    HOSTDEVICE double operator()(double* x) { return x[0]+2*x[1]; }
} simple_function;

TEST_CASE( "fitfunction::PolySingular" , "[fit]" )
{
#define HAS_JACOBIAN
#define HAS_HESSIAN

    using I = simple_function_t;
    using D = double;
    using U = unsigned long long;

    using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;

    U npar = fitfun::function_t::num_parameters;

    SECTION( "num_parameters" )
    {
        REQUIRE(U(fitfun::transform_t::num_parameters) == npar);
#ifdef HAS_JACOBIAN
        REQUIRE(U(fitfun::jacobian_t::num_parameters) == npar);
#endif
    };

    fitfun::function_t ffun;
    std::vector<D> par = ffun.initial_parameters.at(0);

    SECTION( "fit function" )
    {
        REQUIRE(ffun(0.,par.data()) == Approx(0.));
        REQUIRE(ffun(1.,par.data()) == Approx(1.));
        for (U i = 0; i<npar;++i) par[i] *= 1.+1e-3;
        REQUIRE(ffun(0.,par.data()) == Approx(0.));
        REQUIRE(ffun(1.,par.data()) == Approx(1.));
    };

#ifdef HAS_JACOBIAN
    fitfun::jacobian_t fjac;
    SECTION( "fit jacobian" )
    {
        D x=0.2;
        D f0 = ffun(x,par.data());
        for (U i = 0; i<npar;++i)
        {
            par[i]+=1e-6;
            D f1 = ffun(x,par.data());
            REQUIRE((f1-f0)/1e-6 == Approx(fjac(x,par.data(),i)).epsilon(5e-6));
            f0 = f1;
        }

        REQUIRE(ffun(0.,par.data()) == Approx(0.));
        REQUIRE(ffun(1.,par.data()) == Approx(1.));
        for (U i = 0; i<npar;++i) par[i] *= 1.+1e-3;
        REQUIRE(ffun(0.,par.data()) == Approx(0.));
        REQUIRE(ffun(1.,par.data()) == Approx(1.));
    };

#ifdef HAS_HESSIAN
    fitfun::hessian_t fhes;
    SECTION( "fit hessian" )
    {
        D x=0.3;
        D hess=0.;
        std::vector<D> v = {1.0, 1.1,1.2,1.3,1.4,1.5};
        for (U i = 0; i<npar;++i)
        {
            for (U j = 0; j<npar;++j)
            {
                std::vector<D> par = ffun.initial_parameters.at(0);
                D j1 = fjac(x,par.data(),i);
                par[j] += 1e-6;
                D j2 = fjac(x,par.data(),i);
                hess += (j2-j1)/1e-6*v[i]*v[j];
            }
        }
        REQUIRE(fhes(x,v.data(),ffun.initial_parameters.at(0).data()) == Approx(hess).epsilon(5e-6));
    };
#endif
#endif

    SECTION( "fit transform" )
    {
        D wgt=1.;
        D x0 = ffun(0.3,ffun.initial_parameters.at(0).data());
        wgt *= (ffun(0.3+1e-6,ffun.initial_parameters.at(0).data())-x0)/1e-6;
        D x1 = ffun(0.6,par.data());
        wgt *= (ffun(0.6+1e-6,par.data())-x1)/1e-6;
        D x[2] = {x0,x1};
        wgt *= simple_function(x);

        fitfun::transform_t ftra(simple_function);
        for (U i = 0; i<npar;++i)
        {
            ftra.p[0][i] = ffun.initial_parameters.at(0).at(i);
            ftra.p[1][i] = par.at(i);
        }
        x[0]=0.3; x[1] = 0.6;
        REQUIRE(wgt == Approx(ftra(x)));
    };
#undef HAS_JACOBIAN
};

TEST_CASE( "fitfunction::None" , "[fit]" )
{
    using I = simple_function_t;
    using D = double;
    using U = unsigned long long;

    using fitfun = integrators::fitfunctions::None::type<I,D,2>;

    U npar = fitfun::function_t::num_parameters;

    SECTION( "num_parameters" )
    {
        REQUIRE(U(fitfun::transform_t::num_parameters) == npar);
    };
};
