#include "catch.hpp"
#include "qmc.hpp"

#include <complex>
#include <random>
#include <stdexcept> // invalid_argument
#include <limits> // numeric_limits

TEST_CASE( "Qmc Constructor", "[Qmc]" ) {

    integrators::Qmc<double,double> real_integrator;

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

    };

};

TEST_CASE( "Alter Fields", "[Qmc]" ) {

    std::vector<unsigned long long int> v2 = {40,50,60};
    std::vector<unsigned long long int> v3 = {40,50,60};
    std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
    gv[2] = v2;
    gv[3] = v3;

    SECTION( "Check Fields", "[Qmc]" ) {

        integrators::Qmc<double,double> real_integrator;
        real_integrator.minn = 1;
        real_integrator.minm = 2;
        real_integrator.cputhreads = 3;
        real_integrator.generatingvectors = gv;

        REQUIRE( real_integrator.minn == 1 );
        REQUIRE( real_integrator.minm == 2 );
        REQUIRE( real_integrator.cputhreads == 3);
        REQUIRE( real_integrator.generatingvectors.size() == 2 );
        REQUIRE( real_integrator.generatingvectors[2] == v2 );
        REQUIRE( real_integrator.generatingvectors[3] == v3 );


//        Logger logger;
//
//        G randomgenerator;
//
//        U minn;
//        U minm;
//        D epsrel;
//        D epsabs;
//        U maxeval;
//        U maxnperpackage;
//        U maxmperpackage;
//        ErrorMode errormode;
//        U cputhreads;
//        U cudablocks;
//        U cudathreadsperblock;
//        std::set<int> devices;
//        std::map<U,std::vector<U>> generatingvectors;
//        U verbosity;

    };

    SECTION( "Check get_next_n Function", "[Qmc]" ) {

        integrators::Qmc<double,double, unsigned long long int> real_integrator;
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

        real_integrator.generatingvectors[std::numeric_limits<unsigned long long int>::max()] = {1,2,3};
        real_integrator.minn = std::numeric_limits<unsigned long long int>::max();
        // n larger than representable in signed version of 'U' (invalid)
        REQUIRE( real_integrator.minn == std::numeric_limits<unsigned long long int>::max() );
        REQUIRE_THROWS_AS( real_integrator.get_next_n(std::numeric_limits<unsigned long long int>::max()), std::domain_error );

    };
    
};

TEST_CASE( "Exceptions", "[Qmc]" ) {

    std::function<double(double[])> constant_function = [] (double x[]) { return 1; };
    std::function<double(double[])> real_function = [] (double x[]) { return x[0]*x[1]*x[2]; };

    integrators::Qmc<double,double> real_integrator;

    SECTION( "Invalid Dimension", "[Qmc]" ) {

        REQUIRE_THROWS_AS( real_integrator.integrate(constant_function,0) , std::invalid_argument);

    };

    SECTION( "Invalid Number of Random Shifts", "[Qmc]" ) {

        real_integrator.minm = 1;
        REQUIRE_THROWS_AS( real_integrator.integrate(real_function,3) , std::domain_error);

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
        REQUIRE_THROWS_AS( real_integrator.integrate(real_function,3) , std::domain_error);
        
    };
    
};

TEST_CASE( "Integrate", "[Qmc]" ) {

    double eps = 1e-10; // approximate upper bound on integration error we would expect
    double badeps = 0.1; // approximate upper bound on integration error when using very few samples

    integrators::result<double> real_result;
    integrators::result<std::complex<double>> complex_result;

    std::function<double(double[])> univariate_real_function = [] (double x[]) { return x[0]; };
    std::function<std::complex<double>(double[])> univariate_complex_function = [] (double x[]) { return std::complex<double>(x[0],x[0]); };

    std::function<double(double[])> real_function = [] (double x[]) { return x[0]*x[1]; };
    std::function<std::complex<double>(double[])> complex_function = [] (double x[]) { return std::complex<double>(x[0],x[0]*x[1]); };

    integrators::Qmc<double,double> real_integrator;
    real_integrator.minn = 10000;

    integrators::Qmc<std::complex<double>,double> complex_integrator;
    complex_integrator.minn = 10000;

    SECTION( "Real Function (Default Block Size)" )
    {

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (Serial)" )
    {
        real_integrator.cputhreads = 1;

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );
        
    };

    SECTION( "Real Function (Parallel)" )
    {
        real_integrator.cputhreads = 2;

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (Block Size Larger than N)" )
    {
        std::vector<unsigned long long int> v1 = {1};
        std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
        gv[3] = v1;
        real_integrator.generatingvectors = gv;

        real_integrator.minn = 1;
        real_integrator.cputhreads = 5;

        real_result = real_integrator.integrate(univariate_real_function,1);

        REQUIRE( real_integrator.cputhreads == 5 );
        REQUIRE( real_result.integral == Approx(0.5).epsilon(badeps) );
        REQUIRE( real_result.error < badeps );
        
    };

    SECTION( "Complex Function (Default Block Size)" )
    {
        complex_result = complex_integrator.integrate(complex_function,2);

        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(eps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.25).epsilon(eps) );

        REQUIRE( complex_result.error.real() < eps );
        REQUIRE( complex_result.error.imag() < eps );

    };

    SECTION( "Complex Function (Serial)" )
    {
        complex_integrator.cputhreads = 1;

        complex_result = complex_integrator.integrate(complex_function,2);

        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(eps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.25).epsilon(eps) );

        REQUIRE( complex_result.error.real() < eps );
        REQUIRE( complex_result.error.imag() < eps );
        
    };

    SECTION( "Complex Function (Parallel)" )
    {
        complex_integrator.cputhreads = 2;

        complex_result = complex_integrator.integrate(complex_function,2);

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

        complex_integrator.minn = 1;
        complex_integrator.cputhreads = 5;

        complex_result = complex_integrator.integrate(univariate_complex_function,1);

        REQUIRE( complex_integrator.cputhreads == 5 );
        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(badeps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.5).epsilon(badeps) );

        REQUIRE( complex_result.error.real() < badeps );
        REQUIRE( complex_result.error.imag() < badeps );

    };

    SECTION( "Change Seed of Random Number Generator" )
    {

        real_integrator.randomgenerator = std::mt19937_64(1);

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Pick Random Seed for Random Number Generator" )
    {

        std::random_device rd;
        real_integrator.randomgenerator = std::mt19937_64(rd());

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Set cputhreads to zero (sequential)")
    {
        real_integrator.cputhreads = 0;
        complex_integrator.cputhreads = 0;

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

};

TEST_CASE( "Transform Validity", "[Qmc]" ) {

    std::function<double(double[])> throw_function = [] (double x[]) {
        if ( x[0] < 0. ) {
            throw std::invalid_argument("x[0] < 0.");
        } else if ( x[0] > 1. ) {
            throw std::invalid_argument("x[0] > 1.");
        }
        return x[0];
    };

    integrators::Qmc<double,double> real_integrator;
    real_integrator.minn = 10000;

    SECTION( "Check integration parameters x satisfy x >= 0 and x <= 1" )
    {
        REQUIRE_NOTHROW( real_integrator.integrate(throw_function,1));
    };

};

// TODO : Check that we can derive from the class and implement a different integralTransform <--- also make an example
