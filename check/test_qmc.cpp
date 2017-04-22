#include "catch.hpp"
#include "../src/qmc.hpp"

#include <complex>
#include <random>
#include <stdexcept> // invalid_argument

TEST_CASE( "Result Constructor", "[result]") {

    integrators::result<double> result({1.0,2.0});

    SECTION( "Access Fields" )
    {
        REQUIRE( result.integral == Approx(1.0) );
        REQUIRE( result.error == Approx(2.0) );
    };

};

TEST_CASE( "Qmc Constructor", "[Qmc]" ) {

    integrators::Qmc<double,double> real_integrator;

    SECTION( "Check Fields", "[Qmc]" ) {

        REQUIRE( real_integrator.minN > 0 );
        REQUIRE( real_integrator.m > 1 );
        REQUIRE( real_integrator.blockSize > 0);
        REQUIRE( real_integrator.generatingVectors.size() > 0);

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
        real_integrator.minN = 1;
        real_integrator.m = 2;
        real_integrator.blockSize = 3;
        real_integrator.generatingVectors = gv;

        REQUIRE( real_integrator.minN == 1 );
        REQUIRE( real_integrator.m == 2 );
        REQUIRE( real_integrator.blockSize == 3);
        REQUIRE( real_integrator.generatingVectors.size() == 2 );
        REQUIRE( real_integrator.generatingVectors[2] == v2 );
        REQUIRE( real_integrator.generatingVectors[3] == v3 );

    };

    SECTION( "Check getN Function", "[Qmc]" ) {

        integrators::Qmc<double,double> real_integrator;
        real_integrator.minN = 1;
        real_integrator.generatingVectors = gv;

        // minN less than any generating vector
        REQUIRE( real_integrator.minN == 1 );
        REQUIRE( real_integrator.getN() == 2 ); // Increased to smallest generating vector

        real_integrator.minN = 2;
        // minN matches a generating vector
        REQUIRE( real_integrator.minN == 2 );
        REQUIRE( real_integrator.getN() == 2 );

        real_integrator.minN = 4;
        // minN larger than any generating vector
        REQUIRE( real_integrator.minN == 4 );
        REQUIRE( real_integrator.getN() == 3 ); // Decreased to largest generating vector

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

        real_integrator.m = 1;
        REQUIRE_THROWS_AS( real_integrator.integrate(real_function,3) , std::invalid_argument);

    };

    SECTION( "Too Large Dimension", "[Qmc]" ) {

        // Give generating vectors with only 2 dimensions
        std::vector<unsigned long long int> v1 = {40,50};
        std::vector<unsigned long long int> v2 = {40,50};
        std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
        gv[1] = v1;
        gv[2] = v2;

        real_integrator.generatingVectors = gv;

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
    real_integrator.minN = 10000;

    integrators::Qmc<std::complex<double>,double> complex_integrator;
    complex_integrator.minN = 10000;

    SECTION( "Real Function (Default Block Size)" )
    {

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (Serial)" )
    {
        real_integrator.blockSize = 1;

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );
        
    };

    SECTION( "Real Function (Parallel)" )
    {
        real_integrator.blockSize = 2;

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Real Function (Block Size Larger than N)" )
    {
        std::vector<unsigned long long int> v1 = {1};
        std::map< unsigned long long int, std::vector<unsigned long long int> > gv;
        gv[3] = v1;
        real_integrator.generatingVectors = gv;

        real_integrator.blockSize = 5;

        real_result = real_integrator.integrate(univariate_real_function,1);

        REQUIRE( real_integrator.blockSize > real_integrator.getN() );
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
        complex_integrator.blockSize = 1;

        complex_result = complex_integrator.integrate(complex_function,2);

        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(eps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.25).epsilon(eps) );

        REQUIRE( complex_result.error.real() < eps );
        REQUIRE( complex_result.error.imag() < eps );
        
    };

    SECTION( "Complex Function (Parallel)" )
    {
        complex_integrator.blockSize = 2;

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
        complex_integrator.generatingVectors = gv;

        complex_integrator.blockSize = 5;

        complex_result = complex_integrator.integrate(univariate_complex_function,1);

        REQUIRE( complex_integrator.blockSize > complex_integrator.getN() );
        REQUIRE( complex_result.integral.real() == Approx(0.5).epsilon(badeps) );
        REQUIRE( complex_result.integral.imag() == Approx(0.5).epsilon(badeps) );

        REQUIRE( complex_result.error.real() < badeps );
        REQUIRE( complex_result.error.imag() < badeps );

    };

    SECTION( "Change Seed of Random Number Generator" )
    {

        real_integrator.randomGenerator = std::mt19937_64(1);

        real_result = real_integrator.integrate(real_function,2);

        REQUIRE( real_result.integral == Approx(0.25).epsilon(eps) );
        REQUIRE( real_result.error < eps );

    };

    SECTION( "Pick Random Seed for Random Number Generator" )
    {

        std::random_device rd;
        real_integrator.randomGenerator = std::mt19937_64(rd());

        real_result = real_integrator.integrate(real_function,2);

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
    real_integrator.minN = 10000;

    SECTION( "Check integration parameters x satisfy x >= 0 and x <= 1" )
    {
        REQUIRE_NOTHROW( real_integrator.integrate(throw_function,1));
    };

};

// TODO : Check that we can derive from the class and implement a different integralTransform <--- also make an example
