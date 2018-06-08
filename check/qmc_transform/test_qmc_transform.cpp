#include "catch.hpp"
#include "qmc.hpp"

TEST_CASE( "ipow", "[ipow]")
{

    SECTION( "Integer Raised to Unsigned Integer" )
    {
        using D = int;
        using U = unsigned long long int;

        REQUIRE(integrators::transforms::detail::ipow<D,U,0>::value(1) == 1 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,0>::value(3) == 1 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,1>::value(2) == 2 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,1>::value(-2) == -2 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,3>::value(2) == 8 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,3>::value(-2) == -8 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,10>::value(3) == 59049 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,10>::value(-3) == 59049 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,9>::value(4) == 262144 );
        REQUIRE(integrators::transforms::detail::ipow<D,U,9>::value(-4) == -262144 );

    };

    SECTION( "Double Raised to Unsigned Integer" )
    {
        using D = double;
        using U = unsigned long long int;

        REQUIRE(integrators::transforms::detail::ipow<D,U,0>::value(1.) == Approx(1.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,0>::value(3.) == Approx(1.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,1>::value(2.) == Approx(2.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,1>::value(-2.) == Approx(-2.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,3>::value(2.) == Approx(8.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,3>::value(-2.) == Approx(-8.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,10>::value(3.) == Approx(59049.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,10>::value(-3.) == Approx(59049.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,9>::value(4.) == Approx(262144.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,9>::value(-4.) == Approx(-262144.) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,101>::value(-4.) == Approx(-6.42775217703596110e60) );
        REQUIRE(integrators::transforms::detail::ipow<D,U,40>::value(7e-6) == Approx(6.36680576090902799e-207) );
    };

};

TEST_CASE( "Binomial", "[transform]")
{

    SECTION( "Value" )
    {
        using U = unsigned long long int;

        // static_assert used as Binomial should evaluate to an unsigned long long int at compile time
        static_assert(integrators::transforms::detail::Binomial<U,5,2>::value   == 10ull, "Binomial<U,5,2>::value != 10ull");
        static_assert(integrators::transforms::detail::Binomial<U,2,5>::value   == 0ull, "Binomial<U,2,5>::value != 0ull");
        static_assert(integrators::transforms::detail::Binomial<U,15,30>::value == 0ull, "Binomial<U,15,30>::value != 0ull");
        static_assert(integrators::transforms::detail::Binomial<U,30,15>::value == 155117520ull, "Binomial<U,30,15>::value != 155117520ull");
        static_assert(integrators::transforms::detail::Binomial<U,120,15>::value == 4730523156632595024ull, "Binomial<U,120,15>::value != 4730523156632595024ull");
        static_assert(integrators::transforms::detail::Binomial<U,15,120>::value == 0ull, "Binomial<U,15,120>::value != 0ull");
        static_assert(integrators::transforms::detail::Binomial<U,15,15>::value == 1ull, "Binomial<U,15,15>::value != 1ull");
        static_assert(integrators::transforms::detail::Binomial<U,15,0>::value  == 1ull, "Binomial<U,15,0>::value != 1ull");
        static_assert(integrators::transforms::detail::Binomial<U,0,15>::value  == 0ull, "Binomial<U,0,15>::value != 0ull");
        static_assert(integrators::transforms::detail::Binomial<U,0,0>::value   == 1ull, "Binomial<U,0,0>::value != 1ull");
        static_assert(integrators::transforms::detail::Binomial<U,0,1>::value   == 0ull, "Binomial<U,0,1>::value != 0ull");
        static_assert(integrators::transforms::detail::Binomial<U,1,0>::value   == 1ull, "Binomial<U,1,0>::value != 1ull");
    };

};

TEST_CASE( "KorobovCoefficient", "[transform]")
{

    // TODO - Integer Coefficients
    SECTION( "Integer Coefficients" )
    {
        using D = long long int;
        using U = unsigned long long int;

        // static_assert as KorobovCoefficient should evaluate to an int at compile time
        // r=1, (3 - 2 x) x^2
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,1>::value == 3, "KorobovCoefficient<D,U,0,1,1>::value != 3");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,1,1,1>::value == -2, "KorobovCoefficient<D,U,1,1,1>::value != -2");

        // r=2, x^3 (10 + x (-15 + 6 x))
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,2>::value == 10,  "KorobovCoefficient<D,U,0,2,2>::value != 10");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,1,2,2>::value == -15, "KorobovCoefficient<D,U,1,2,2>::value != -15");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,2,2,2>::value == 6,   "KorobovCoefficient<D,U,2,2,2>::value != 6");

        // r=3, x^4 (35 + x (-84 + (70 - 20 x) x))
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,3>::value == 35,  "KorobovCoefficient<D,U,0,3,3>::value != 35");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,1,3,3>::value == -84, "KorobovCoefficient<D,U,1,3,3>::value != -84");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,2,3,3>::value == 70,  "KorobovCoefficient<D,U,2,3,3>::value != 70");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,3,3,3>::value == -20, "KorobovCoefficient<D,U,3,3,3>::value != -20");

        // r=4, x^5 (126 + x (-420 + x (540 + x (-315 + 70 x))))
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,4>::value == 126,  "KorobovCoefficient<D,U,0,2,2>::value != 126");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,1,4,4>::value == -420, "KorobovCoefficient<D,U,1,2,2>::value != -420");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,2,4,4>::value == 540,  "KorobovCoefficient<D,U,2,2,2>::value != 540");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,3,4,4>::value == -315, "KorobovCoefficient<D,U,3,2,2>::value != -315");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,4,4,4>::value == 70,   "KorobovCoefficient<D,U,4,2,2>::value != -70");

        // r=8, x^9 (24310+x (-175032+x (556920+x (-1021020+x (1178100+x (-875160+x (408408+x (-109395+12870 x))))))))
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,0,8,8>::value == 24310,    "KorobovCoefficient<D,U,0,8,8>::value != 24310");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,1,8,8>::value == -175032,  "KorobovCoefficient<D,U,1,8,8>::value != 175032");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,2,8,8>::value == 556920,   "KorobovCoefficient<D,U,2,8,8>::value != 556920");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,3,8,8>::value == -1021020, "KorobovCoefficient<D,U,3,8,8>::value != -1021020");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,4,8,8>::value == 1178100,  "KorobovCoefficient<D,U,4,8,8>::value != 1178100");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,5,8,8>::value == -875160,  "KorobovCoefficient<D,U,5,8,8>::value != -875160");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,6,8,8>::value == 408408,   "KorobovCoefficient<D,U,6,8,8>::value != 408408");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,7,8,8>::value == -109395,  "KorobovCoefficient<D,U,7,8,8>::value != -109395");
        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,8,8,8>::value == 12870,    "KorobovCoefficient<D,U,8,8,8>::value != 12870");

        // Uneven weight
//        static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,5,6,7>::value == -90090, "KorobovCoefficient<U,5,6,7>::value != -90090"); // TODO - overflow?
     };

    SECTION( "Double Coefficients" )
    {
        using D = double;
        using U = unsigned long long int;

        // could use static_assert as KorobovCoefficient should evaluate to a double at compile time, but want to use Approx for comparing floating point

        double korobov_coefficient; // workaround catch issue

        // r=1, (3 - 2 x) x^2
        korobov_coefficient = integrators::transforms::detail::KorobovCoefficient<D,U,0,1,1>::value;
        REQUIRE( korobov_coefficient== Approx(3.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,1,1,1>::value;
        REQUIRE( korobov_coefficient == Approx(-2.) );

        // r=2, x^3 (10 + x (-15 + 6 x))
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,0,2,2>::value;
        REQUIRE( korobov_coefficient == Approx(10.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,1,2,2>::value;
        REQUIRE( korobov_coefficient == Approx(-15.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,2,2,2>::value;
        REQUIRE( korobov_coefficient == Approx(6.) );

        // r=3, x^4 (35 + x (-84 + (70 - 20 x) x))
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,0,3,3>::value;
        REQUIRE( korobov_coefficient == Approx(35.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,1,3,3>::value;
        REQUIRE( korobov_coefficient == Approx(-84.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,2,3,3>::value;
        REQUIRE( korobov_coefficient == Approx(70.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,3,3,3>::value;
        REQUIRE( korobov_coefficient == Approx(-20.) );

        // r=4, x^5 (126 + x (-420 + x (540 + x (-315 + 70 x))))
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,0,4,4>::value;
        REQUIRE( korobov_coefficient == Approx(126.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,1,4,4>::value;
        REQUIRE( korobov_coefficient == Approx(-420.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,2,4,4>::value;
        REQUIRE( korobov_coefficient == Approx(540.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,3,4,4>::value;
        REQUIRE( korobov_coefficient == Approx(-315.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,4,4,4>::value;
        REQUIRE( korobov_coefficient == Approx(70.) );

        // r=8, x^9 (24310+x (-175032+x (556920+x (-1021020+x (1178100+x (-875160+x (408408+x (-109395+12870 x))))))))
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,0,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(24310.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,1,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(-175032.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,2,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(556920.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,3,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(-1021020.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,4,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(1178100.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,5,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(-875160.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,6,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(408408.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,7,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(-109395.) );
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,8,8,8>::value;
        REQUIRE( korobov_coefficient == Approx(12870.) );

        // Uneven weight
        korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,5,6,7>::value;
        REQUIRE( korobov_coefficient == Approx(-90090.) );
    };

};

//TEST_CASE( "Result Constructor", "[result]") {
//
//    integrators::result<double,int> result({1.0,2.0,-3,-4});
//
//    SECTION( "Access Fields" )
//    {
//        REQUIRE( result.integral == Approx(1.0) );
//        REQUIRE( result.error == Approx(2.0) );
//        REQUIRE( result.n == -3 );
//        REQUIRE( result.m == -4 );
//    };
//
//};
