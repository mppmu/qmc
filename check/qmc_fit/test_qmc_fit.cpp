#include "catch.hpp"
#include "qmc.hpp"

#ifdef __CUDACC__
#include <thrust/complex.h>
using thrust::complex;
#define HOSTDEVICE __host__ __device__
#else
using std::complex;
#include <complex>
#define HOSTDEVICE
#endif


#include<iostream>

struct simple_function_t {
    const unsigned long long int number_of_integration_variables = 2;
    HOSTDEVICE double operator()(double* x) { return x[0]+2*x[1]; }
} simple_function;

struct const_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE double operator()(double* x) { return 2. ; }
} const_function;

struct poly_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    const double a=0.6;
    // expexted fit parameters x0: {arbitrary, arbitrary, 0, 0 a, 1-a}
    HOSTDEVICE double operator()(double* x) { return  1./sqrt(a*a+4.*x[0]*(1.-a));}
} poly_function;

struct test_function_t {
    const unsigned long long int number_of_integration_variables = 2;
    const double a=0.2;
    // expexted fit parameters x0: {1.07, arbitrary, 0.3, 0, 0.7, 0}
    // expexted fit parameters x1: {arbitrary, arbitrary, 0, 0, a, 1-a }
    HOSTDEVICE double operator()(double* x) { return  (0.7142857142857143*(1.+(0.728-x[0])/sqrt(-2.996*x[0]+(0.77+x[0])*(0.77+x[0]))))/sqrt(a*a+4.*x[1]*(1.-a)); }
} test_function;

struct test_function2_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE double operator()(double* x) { return  1./(1-x[0]*std::sin(3.*x[0])); }
} test_function2;

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
    }

    fitfun::function_t ffun;
    std::vector<D> par = ffun.initial_parameters.at(0);

    SECTION( "fit function" )
    {
        REQUIRE(ffun(0.,par.data()) == Approx(0.));
        REQUIRE(ffun(1.,par.data()) == Approx(1.));
        for (U i = 0; i<npar;++i) par[i] *= 1.+1e-3;
        REQUIRE(ffun(0.,par.data()) == Approx(0.));
        REQUIRE(ffun(1.,par.data()) == Approx(1.));
    }

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
    }

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
    }
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
    }
#undef HAS_JACOBIAN
}

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
    }
}

TEST_CASE( "chisq fit" , "[fit]" )
{
    using I = test_function_t;
    using D = double;
    using U = unsigned long long;

    SECTION( "constant integrand" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        qmc.fitmaxiter=400;
        qmc.fitxtol=1e-4;
        qmc.verbosity=3;
//        qmc.evaluateminn=10;
        auto fitted_function = qmc.fit(const_function);
        REQUIRE(fitted_function.p[0][2] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(0.).margin(5e-3));
    }
    SECTION( "polynomial inv. CDF" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        qmc.fitmaxiter=400;
        qmc.fitxtol=1e-4;
        qmc.verbosity=3;
        auto fitted_function = qmc.fit(poly_function);
        REQUIRE(fitted_function.p[0][2] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(0.4).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(0.0).margin(5e-3));
    }

    SECTION( "integrand matching fit function" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        qmc.fitmaxiter=400;
        qmc.fitxtol=1e-4;
        qmc.verbosity=3;
        auto fitted_function = qmc.fit(test_function);
        REQUIRE(fitted_function.p[0][0] == Approx(1.07).margin(5e-3));
        REQUIRE(fitted_function.p[0][2] == Approx(0.3).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(0.0).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(0.).margin(5e-3));

        REQUIRE(fitted_function.p[1][2] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[1][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[1][4] == Approx(0.8).margin(5e-3));
        REQUIRE(fitted_function.p[1][5] == Approx(0.0).margin(5e-3));
    }

    SECTION( "integrand not matching fit function" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        qmc.verbosity=3;
        qmc.fitxtol=1e-6;
        qmc.fitmaxiter=400;
        //qmc.fitxtol=1e-6;
        auto fitted_function = qmc.fit(test_function2);
        REQUIRE(fitted_function.p[0][0] == Approx(2.12682).margin(5e-3));
        REQUIRE(fitted_function.p[0][2] == Approx(2.12684).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(0.700739).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(-2.15106).margin(5e-3));
    }
}

