#include "catch.hpp"
#include "qmc.hpp"

#include <cmath> // std::nan
#include <random>
#include <stdexcept> // invalid_argument
#include <limits> // numeric_limits
#include <sstream> // ostringstream
#include <string> // to_string
#include <random> // mt19937_64
#include <utility> // pair
#include <vector>
#include <map>

#ifdef __CUDACC__
#include <thrust/complex.h>
using thrust::complex;
#define HOSTDEVICE __host__ __device__
#else
using std::complex;
#include <complex>
#define HOSTDEVICE
#endif

struct zero_dim_function_t {
    const unsigned long long int number_of_integration_variables = 0;
    HOSTDEVICE double operator()(double x[]) { return 1; }
} zero_dim_function;

struct too_many_dim_function_t {
    const unsigned long long int number_of_integration_variables = 4;
    HOSTDEVICE double operator()(double x[]) { return 1; }
} too_many_dim_function;

struct constant_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE double operator()(double x[]) { return 1; }
} constant_function;

struct multivariate_linear_function_t {
    const unsigned long long int number_of_integration_variables = 3;
    HOSTDEVICE double operator()(double x[]) { return x[0]*x[1]*x[2]; }
} multivariate_linear_function;

struct nan_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE double operator()(double x[]) {
        if ( x[0] < 0. || x[0] > 1. )
            return std::nan("");
        else
            return x[0];
    };
} nan_function;

struct univariate_real_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE double operator()(double x[]) { return x[0]; }
} univariate_real_function;

struct univariate_complex_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE complex<double> operator()(double x[]) { return complex<double>(x[0],x[0]); }
} univariate_complex_function;

struct real_function_t {
    const unsigned long long int number_of_integration_variables = 2;
    HOSTDEVICE double operator()(double x[]) { return x[0]*x[1]; }
} real_function;

struct complex_function_t {
    const unsigned long long int number_of_integration_variables = 2;
    HOSTDEVICE complex<double> operator()(double x[]) { return complex<double>(x[0],x[0]*x[1]); };

} complex_function;

struct poly_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    const double a=0.6;
    // expexted fit parameters x0: {arbitrary, arbitrary, 0, 0 a, 1-a}
    HOSTDEVICE double operator()(double* x) { return  1./sqrt(a*a+4.*x[0]*(1.-a));}
} poly_function;

struct const_function_t {
    const unsigned long long int number_of_integration_variables = 1;
    HOSTDEVICE double operator()(double* x) { return 2. ; }
} const_function;

struct test_function_t {
    const unsigned long long int number_of_integration_variables = 2;
    const double a=0.2;
    // expexted fit parameters x0: {1.07, arbitrary, 0.3, 0, 0.7, 0}
    // expexted fit parameters x1: {arbitrary, arbitrary, 0, 0, a, 1-a }
    HOSTDEVICE double operator()(double* x) { return  (0.7142857142857143*(1.+(0.728-x[0])/sqrt(-2.996*x[0]+(0.77+x[0])*(0.77+x[0]))))/sqrt(a*a+4.*x[1]*(1.-a)); }
} test_function;

struct test_function2_t {
    const unsigned long long int number_of_integration_variables = 1;
    // inverse CDF: (x + 1/3 x Cos[3 x] - 1/9 Sin[3 x])
    HOSTDEVICE double operator()(double* x) { return  1./(1-x[0]*std::sin(3.*x[0])); }
} test_function2;

TEST_CASE( "Qmc Constructor", "[Qmc]" ) {

    integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;

    SECTION( "Check Fields", "[Qmc]" ) {

        std::uniform_real_distribution<double> uniform_distribution(0,1);

        //        Logger logger; // checked later
        // Check a few random samples
        for(int i = 0; i < 100; i++)
        {
            double random_sample = uniform_distribution( real_integrator.randomgenerator );
            REQUIRE( random_sample >= 0 );
            REQUIRE( random_sample <= 1 );
        }
        REQUIRE( real_integrator.minn > 0 );
        REQUIRE( real_integrator.minm > 1 ); // can not calculate variance if minm <= 1
        REQUIRE( real_integrator.epsrel >= 0 );
        REQUIRE( real_integrator.epsabs >= 0 );
        REQUIRE( real_integrator.maxeval >= 0 );
        REQUIRE( real_integrator.maxnperpackage > 0 );
        REQUIRE( real_integrator.maxmperpackage > 0 );
        REQUIRE( real_integrator.errormode > 0 ); // ErrorMode starts at 1
        REQUIRE( real_integrator.cputhreads > 0 );
        REQUIRE( real_integrator.cudablocks > 0 );
        REQUIRE( real_integrator.cudathreadsperblock > 0 );
        REQUIRE( real_integrator.devices.size() > 0 );
        REQUIRE( real_integrator.generatingvectors.size() > 0 );
        REQUIRE( real_integrator.verbosity >= 0 );
        REQUIRE( real_integrator.evaluateminn >= 0);
        REQUIRE( real_integrator.fitstepsize > 0);
        REQUIRE( real_integrator.fitmaxiter > 0);
        REQUIRE( real_integrator.fitxtol >= 0);
        REQUIRE( real_integrator.fitgtol >= 0);
        REQUIRE( real_integrator.fitftol >= 0);
//        REQUIRE( fitparametersgsl == ???); // Not checked

    };

};

TEST_CASE( "Alter Fields", "[Qmc]" ) {

    std::vector<unsigned long long int> v2 = {40,50,60};
    std::vector<unsigned long long int> v3 = {40,50,60};
    std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
    gv[2] = v2;
    gv[3] = v3;

    SECTION( "Check Fields", "[Qmc]" ) {

        integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
        std::ostringstream stream; real_integrator.logger = integrators::Logger(stream);
        real_integrator.randomgenerator = std::mt19937_64( std::random_device{}() );
        real_integrator.minn = 1;
        real_integrator.minm = 2;
        real_integrator.epsrel = 1.;
        real_integrator.epsabs = 1.;
        real_integrator.maxeval = 1;
        real_integrator.maxnperpackage = 1;
        real_integrator.maxmperpackage = 1;
        real_integrator.errormode = integrators::ErrorMode::all;
        real_integrator.cputhreads = 1;
        real_integrator.cudablocks = 1;
        real_integrator.cudathreadsperblock = 1;
        real_integrator.devices = {1};
        real_integrator.generatingvectors = gv;
        real_integrator.verbosity = 0;
        real_integrator.evaluateminn = 1;
        real_integrator.fitstepsize = 1;
        real_integrator.fitmaxiter =1;
        real_integrator.fitxtol = 2.;
        real_integrator.fitgtol = 2.;
        real_integrator.fitftol = 2.;

        REQUIRE( real_integrator.minn == 1 );
        REQUIRE( real_integrator.minm == 2 );
        REQUIRE( real_integrator.epsrel == Approx(1.) );
        REQUIRE( real_integrator.epsabs == Approx(1.) );
        REQUIRE( real_integrator.maxeval == 1 );
        REQUIRE( real_integrator.maxnperpackage == 1 );
        REQUIRE( real_integrator.maxmperpackage == 1 );
        REQUIRE( real_integrator.errormode == integrators::ErrorMode::all );
        REQUIRE( real_integrator.cputhreads == 1 );
        REQUIRE( real_integrator.cudablocks == 1 );
        REQUIRE( real_integrator.cudathreadsperblock == 1 );
        REQUIRE( real_integrator.devices.size() == 1 );
        REQUIRE( real_integrator.generatingvectors.size() == 2 );
        REQUIRE( real_integrator.generatingvectors[2] == v2 );
        REQUIRE( real_integrator.generatingvectors[3] == v3 );
        REQUIRE( real_integrator.verbosity == 0  );
        REQUIRE( real_integrator.evaluateminn == 1 );
        REQUIRE( real_integrator.fitstepsize == 1 );
        REQUIRE( real_integrator.fitmaxiter == 1 );
        REQUIRE( real_integrator.fitxtol == Approx(2.) );
        REQUIRE( real_integrator.fitgtol == Approx(2.) );
        REQUIRE( real_integrator.fitftol == Approx(2.) );

    };

    SECTION( "Check get_next_n Function", "[Qmc]" ) {

        integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
        real_integrator.minn = 1;
        real_integrator.generatingvectors = gv;

        // minn less than any generating vector
        REQUIRE( real_integrator.minn == 1 );
        REQUIRE( real_integrator.get_next_n(1) == 2 ); // Increased to smallest generating vector

        real_integrator.minn = 2;
        // minn matches a generating vector
        REQUIRE( real_integrator.minn == 2 );
        REQUIRE( real_integrator.get_next_n(2) == 2 );

        real_integrator.minn = 4;
        // minn larger than any generating vector
        REQUIRE( real_integrator.minn == 4 );
        REQUIRE( real_integrator.get_next_n(4) == 3 ); // fall back to largest available generating vector

        // n larger than representable in signed version of 'U' (invalid)
        real_integrator.generatingvectors[std::numeric_limits<unsigned long long int>::max()] = {1,2,3};
        real_integrator.minn = std::numeric_limits<unsigned long long int>::max();
        REQUIRE( real_integrator.minn == std::numeric_limits<unsigned long long int>::max() );
        REQUIRE_THROWS_AS( real_integrator.get_next_n(std::numeric_limits<unsigned long long int>::max()), std::domain_error );

        // n larger than the largest finite value representable by the mantissa of float
        integrators::Qmc<float, float,3,integrators::transforms::Korobov<3>::type> float_integrator;
        float_integrator.generatingvectors = { { std::numeric_limits<long long int>::max()-1, {1,2,3} } };
        REQUIRE_THROWS_AS( float_integrator.get_next_n(1), std::domain_error );

    };
    
};

TEST_CASE( "Exceptions", "[Qmc]" ) {

    integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
    std::ostringstream real_stream; real_integrator.logger = integrators::Logger(real_stream);

    SECTION( "Invalid Dimension", "[Qmc]" ) {

        REQUIRE_THROWS_AS( real_integrator.integrate(zero_dim_function) , std::invalid_argument);
        
    };
    
    SECTION( "Invalid Dimension", "[Qmc]" ) {
        
        REQUIRE_THROWS_AS( real_integrator.evaluate(zero_dim_function) , std::invalid_argument);
        
    };

    SECTION( "Invalid Number of Random Shifts", "[Qmc]" ) {

        real_integrator.minm = 1;
        REQUIRE_THROWS_AS( real_integrator.integrate(multivariate_linear_function) , std::domain_error);

    };

    SECTION( "Invalid Number of Random Shifts Per Package", "[Qmc]" ) {

        real_integrator.maxmperpackage = 1;
        REQUIRE_THROWS_AS( real_integrator.integrate(multivariate_linear_function) , std::domain_error);

    };

    SECTION( "Invalid Number of Points Per Package", "[Qmc]" ) {

        real_integrator.maxnperpackage = 0;
        REQUIRE_THROWS_AS( real_integrator.integrate(multivariate_linear_function) , std::domain_error);

    };

    SECTION( "Too Large Dimension", "[Qmc]" ) {

        // Give generating vectors with only 2 dimensions
        std::vector<unsigned long long int> v1 = {40,50};
        std::vector<unsigned long long int> v2 = {40,50};
        std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
        gv[1] = v1;
        gv[2] = v2;

        real_integrator.generatingvectors = gv;

        // Call integrate on function with 3 dimensions
        REQUIRE_THROWS_AS( real_integrator.integrate(multivariate_linear_function) , std::domain_error);
        
    };

    SECTION ( "number_of_integration_variables > M", "[Qmc]")
    {
        REQUIRE_THROWS_AS( real_integrator.integrate(too_many_dim_function), std::invalid_argument );
    };

    SECTION ( "number_of_integration_variables > M", "[Qmc]")
    {
        REQUIRE_THROWS_AS( real_integrator.evaluate(too_many_dim_function), std::invalid_argument );
    };

    SECTION( "Invalid Number of CPU Threads", "[Qmc]")
    {

        real_integrator.cputhreads = 0;
        REQUIRE_THROWS_AS( real_integrator.integrate(multivariate_linear_function) , std::domain_error );

    };

    SECTION( "Invalid Number of CPU Threads", "[Qmc]")
    {

        real_integrator.cputhreads = 0;
        REQUIRE_THROWS_AS( real_integrator.evaluate(multivariate_linear_function) , std::domain_error );

    };

#ifndef __CUDACC__
    SECTION( "Device set to GPU but CUDA Disabled in sample function", "[Qmc]") {

        real_integrator.devices = {1}; // gpu
        real_integrator.evaluateminn = 0; // disable fitting
        REQUIRE_THROWS_AS( real_integrator.integrate(multivariate_linear_function), std::invalid_argument);

    }
#endif

#ifndef __CUDACC__
    SECTION( "Device set to GPU but CUDA Disabled in evaluate function", "[Qmc]") {

        real_integrator.devices = {1}; // gpu
        REQUIRE_THROWS_AS( real_integrator.evaluate(multivariate_linear_function), std::invalid_argument);

    }
#endif

};

TEST_CASE( "Transform Validity", "[Qmc]" ) {

    integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
    std::ostringstream real_stream; real_integrator.logger = integrators::Logger(real_stream);
    real_integrator.minn = 10000;

    SECTION( "Check integration parameters x satisfy x >= 0 and x <= 1" )
    {
        REQUIRE( !std::isnan(real_integrator.integrate(nan_function).integral) );
    };

};

TEST_CASE( "Integrate", "[Qmc]" ) {

    double eps = 1e-6; // approximate upper bound on integration error we would expect
    double badeps = 0.1; // approximate upper bound on integration error when using very few samples

    integrators::result<double> real_result;
    integrators::result<complex<double>> complex_result;

    integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
    std::ostringstream real_stream; real_integrator.logger = integrators::Logger(real_stream);
    real_integrator.minn = 10000;
    real_integrator.verbosity = 3;

    integrators::Qmc<complex<double>,double,3,integrators::transforms::Korobov<3>::type> complex_integrator;
    std::ostringstream complex_stream; complex_integrator.logger = integrators::Logger(complex_stream);
    complex_integrator.minn = 10000;
    complex_integrator.verbosity = 3;

    SECTION( "Real Function (Default Block Size)" )
    {

        real_result = real_integrator.integrate(real_function);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (Serial)" )
    {
        real_integrator.cputhreads = 1;

        real_result = real_integrator.integrate(real_function);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );
        
    };

    SECTION( "Real Function (Parallel)" )
    {
        real_integrator.cputhreads = 2;

        real_result = real_integrator.integrate(real_function);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (Block Size Larger than N)" )
    {
        std::vector<unsigned long long int> v1 = {1};
        std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
        gv[3] = v1;
        real_integrator.generatingvectors = gv;

        real_integrator.evaluateminn = 0; // no fitting because there are too few points
        real_integrator.minn = 1;
        real_integrator.cputhreads = 5;

        real_result = real_integrator.integrate(univariate_real_function);

        REQUIRE( real_integrator.cputhreads == 5 );
        REQUIRE( real_result.integral == Approx(0.5).epsilon(badeps) );
        REQUIRE( real_result.error < badeps );
        
    };

    SECTION( "Complex Function (Default Block Size)" )
    {
        complex_result = complex_integrator.integrate(complex_function);

        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(eps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.25).epsilon(eps) );

        REQUIRE( complex_result.error.real() < eps );
        REQUIRE( complex_result.error.imag() < eps );

    };

    SECTION( "Complex Function (Serial)" )
    {
        complex_integrator.cputhreads = 1;

        complex_result = complex_integrator.integrate(complex_function);

        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(eps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.25).epsilon(eps) );

        REQUIRE( complex_result.error.real() < eps );
        REQUIRE( complex_result.error.imag() < eps );
        
    };

    SECTION( "Complex Function (Parallel)" )
    {
        complex_integrator.cputhreads = 2;

        complex_result = complex_integrator.integrate(complex_function);

        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(eps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.25).epsilon(eps) );

        REQUIRE( complex_result.error.real() < eps );
        REQUIRE( complex_result.error.imag() < eps );
        
    };

    SECTION( "Complex Function (Block Size Larger than N)" )
    {
        std::vector<unsigned long long int> v1 = {1};
        std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
        gv[3] = v1;
        complex_integrator.generatingvectors = gv;

        complex_integrator.evaluateminn = 0; // no fitting because there are too few points
        complex_integrator.minn = 1;
        complex_integrator.cputhreads = 5;

        complex_result = complex_integrator.integrate(univariate_complex_function);

        REQUIRE( complex_integrator.cputhreads == 5 );
        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(badeps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.5).epsilon(badeps) );

        REQUIRE( complex_result.error.real() < badeps );
        REQUIRE( complex_result.error.imag() < badeps );

    };

    SECTION( "Change Seed of Random Number Generator" )
    {

        real_integrator.randomgenerator = std::mt19937_64(1);

        real_result = real_integrator.integrate(real_function);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Pick Random Seed for Random Number Generator" )
    {

        std::random_device rd;
        real_integrator.randomgenerator = std::mt19937_64(rd());

        real_result = real_integrator.integrate(real_function);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (apply_fit && transform)" )
    {

        real_result = real_integrator.integrate(constant_function);

        REQUIRE( real_result.integral == Approx(1.).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (apply_fit && !transform)" )
    {

        real_result = real_integrator.integrate(constant_function);

        REQUIRE( real_result.integral == Approx(1.).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (!apply_fit && transform)" )
    {

        real_integrator.evaluateminn = 0;
        real_result = real_integrator.integrate(constant_function);

        REQUIRE( real_result.integral == Approx(1.).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (!apply_fit && !transform)" )
    {

        real_integrator.evaluateminn = 0;
        real_result = real_integrator.integrate(constant_function);

        REQUIRE( real_result.integral == Approx(1.).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

};

TEST_CASE( "Integrate Monte-Carlo Scaling", "[Qmc]" ) {
    double eps = 1e-6; // approximate upper bound on integration error we would expect

    integrators::result<double> real_result;

    integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
    std::ostringstream real_stream; real_integrator.logger = integrators::Logger(real_stream);
    real_integrator.verbosity = 3;

    SECTION( "Switch to Monte-Carlo scaling due to available lattice sizes" )
    {

        std::map<unsigned long long int,std::vector<unsigned long long int>> gv;
        gv[1021] = {1,374,421};

        real_integrator.evaluateminn = 0;
        real_integrator.generatingvectors = gv;
        real_integrator.epsrel = 1e-10;
        real_integrator.epsabs = 1e-10;
        real_integrator.maxeval = 1000000;

        real_result = real_integrator.integrate(multivariate_linear_function);

        REQUIRE( real_result.integral == Approx(0.125).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Reduce Lattice Size due to low maxeval" )
    {

        std::map<unsigned long long int,std::vector<unsigned long long int>> gv;
        gv[1021] = {1,374,421};
        gv[2147483647] = {1,367499618,943314825}; // too big to use due to maxeval

        real_integrator.minn = 1;
        real_integrator.evaluateminn = 0;
        real_integrator.generatingvectors = gv;
        real_integrator.epsrel = 1e-10;
        real_integrator.epsabs = 1e-10;
        real_integrator.maxeval = 130688;

        real_result = real_integrator.integrate(multivariate_linear_function);

        REQUIRE( real_result.integral == Approx(0.125).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };
};

TEST_CASE( "Evaluate", "[qmc]") {
    
    using U = unsigned long long int;
    
    integrators::samples<double,double> real_samples;
    integrators::samples<complex<double>,double> complex_samples;

    U dim = 2;
    U n = 5;
    std::vector<U> z = {1,2};
    std::map<U,std::vector<U>> test_generating_vectors;
    test_generating_vectors.insert(std::pair<U,std::vector<U>>(n,z));
    
    integrators::Qmc<double,double,3,integrators::transforms::Korobov<3>::type> real_integrator;
    std::ostringstream real_stream; real_integrator.logger = integrators::Logger(real_stream);
    real_integrator.evaluateminn = 100;
    real_integrator.verbosity = 3;
    real_integrator.generatingvectors = test_generating_vectors;
    
    integrators::Qmc<complex<double>,double,3,integrators::transforms::Korobov<3>::type> complex_integrator;
    std::ostringstream complex_stream; complex_integrator.logger = integrators::Logger(complex_stream);
    real_integrator.evaluateminn = 100;
    complex_integrator.verbosity = 3;
    complex_integrator.generatingvectors = test_generating_vectors;
    
    SECTION( "Real Function (Serial)" )
    {
        real_integrator.cputhreads = 1;
        
        real_samples = real_integrator.evaluate(real_function);
        
        for(U i = 0; i < dim; i++)
        {
            REQUIRE( real_samples.z.at(i) == z.at(i) );
        }
        REQUIRE( real_samples.d.size() == dim);
        REQUIRE( real_samples.r.size() == n);
        REQUIRE( real_samples.n == n );
    };
    
    SECTION( "Real Function (Parallel)" )
    {
        real_integrator.cputhreads = 2;
        
        real_samples = real_integrator.evaluate(real_function);
        
        for(U i = 0; i < dim; i++)
        {
            REQUIRE( real_samples.z.at(i) == z.at(i) );
        }
        REQUIRE( real_samples.d.size() == dim);
        REQUIRE( real_samples.r.size() == n);
        REQUIRE( real_samples.n == n );
        
    };
    
    SECTION( "Complex Function (Serial)" )
    {
        complex_integrator.cputhreads = 1;
        
        complex_samples = complex_integrator.evaluate(complex_function);
        
        for(U i = 0; i < dim; i++)
        {
            REQUIRE( complex_samples.z.at(i) == z.at(i) );
        }
        REQUIRE( complex_samples.d.size() == dim);
        REQUIRE( complex_samples.r.size() == n);
        REQUIRE( complex_samples.n == n );
        
    };
    
    SECTION( "Complex Function (Parallel)" )
    {
        complex_integrator.cputhreads = 2;
        
        complex_samples = complex_integrator.evaluate(complex_function);
        
        for(U i = 0; i < dim; i++)
        {
            REQUIRE( complex_samples.z.at(i) == z.at(i) );
        }
        REQUIRE( complex_samples.d.size() == dim);
        REQUIRE( complex_samples.r.size() == n);
        REQUIRE( complex_samples.n == n );
        
    };
    
};

TEST_CASE( "Fitting (Chi-sq)" , "[fit]" )
{
    using I = test_function_t;
    using D = double;
    using U = unsigned long long;
    
    SECTION( "constant integrand" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        auto fitted_function = qmc.fit(const_function);
        REQUIRE(fitted_function.p[0][2] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(1.).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(0.).margin(5e-3));
    };
    
    SECTION( "polynomial inv. CDF" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        auto fitted_function = qmc.fit(poly_function);
        REQUIRE(fitted_function.p[0][2] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(0.6).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(0.4).margin(5e-3));
    };
    
    SECTION( "integrand matching fit function" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        qmc.randomgenerator.seed(1);
        auto fitted_function = qmc.fit(test_function);
        REQUIRE(fitted_function.p[0][0] == Approx(1.07).margin(5e-3));
        REQUIRE(fitted_function.p[0][2] == Approx(0.3).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(0.7).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(0.).margin(5e-3));
        
        REQUIRE(fitted_function.p[1][2] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[1][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[1][4] == Approx(0.2).margin(5e-3));
        REQUIRE(fitted_function.p[1][5] == Approx(0.8).margin(5e-3));
    };
    
    SECTION( "integrand not matching fit function" )
    {
        using fitfun = integrators::fitfunctions::PolySingular::type<I,D,2>;
        integrators::Qmc<double,double,2,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> qmc;
        std::ostringstream qmc_stream; qmc.logger = integrators::Logger(qmc_stream);
        qmc.randomgenerator.seed(1);
        qmc.verbosity=3;
        auto fitted_function = qmc.fit(test_function2);
        // reasonable results of fit, but values might change when modifying fit procedure
        REQUIRE(fitted_function.p[0][0] == Approx(1.98118).margin(5e-3));
        REQUIRE(fitted_function.p[0][2] == Approx(1.58157).margin(5e-3));
        REQUIRE(fitted_function.p[0][3] == Approx(0.).margin(5e-3));
        REQUIRE(fitted_function.p[0][4] == Approx(1.04626).margin(5e-3));
        REQUIRE(fitted_function.p[0][5] == Approx(-2.0317).margin(5e-3));
    };
};
