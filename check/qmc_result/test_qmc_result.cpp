#include "catch.hpp"
#include "qmc.hpp"

TEST_CASE( "Result Constructor (default)", "[result]") {

    integrators::result<double> result({1.0,2.0,3,4,5,6});

    SECTION( "Access Fields" )
    {
        REQUIRE( result.integral == Approx(1.0) );
        REQUIRE( result.error == Approx(2.0) );
        REQUIRE( result.n == 3 );
        REQUIRE( result.m == 4 );
        REQUIRE( result.iterations == 5 );
        REQUIRE( result.evaluations == 6 );
    };

};
