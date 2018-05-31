#include "catch.hpp"
#include "qmc.hpp"

TEST_CASE( "Result Constructor (default)", "[result]") {

    integrators::result<double> result({1.0,2.0,3,4});

    SECTION( "Access Fields" )
    {
        REQUIRE( result.integral == Approx(1.0) );
        REQUIRE( result.error == Approx(2.0) );
        REQUIRE( result.n == 3 );
        REQUIRE( result.m == 4 );
    };

};

TEST_CASE( "Result Constructor", "[result]") {

    integrators::result<double,int> result({1.0,2.0,-3,-4});

    SECTION( "Access Fields" )
    {
        REQUIRE( result.integral == Approx(1.0) );
        REQUIRE( result.error == Approx(2.0) );
        REQUIRE( result.n == -3 );
        REQUIRE( result.m == -4 );
    };

};
