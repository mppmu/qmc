#include "catch.hpp"
#include "qmc.hpp"

#include <string> // stod

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

        // Uneven weight (not used by program)
        // static_assert(integrators::transforms::detail::KorobovCoefficient<D,U,5,6,7>::value == -90090, "KorobovCoefficient<U,5,6,7>::value != -90090");
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

        // Uneven weight (not used by program)
        // korobov_coefficient =integrators::transforms::detail::KorobovCoefficient<D,U,5,6,7>::value;
        //  REQUIRE( korobov_coefficient == Approx(-90090.) );
    };

};

TEST_CASE( "KorobovTerm", "[transform]")
{
    using D = double;
    using U = unsigned long long int;

    U dim = 9;
    D x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    D y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    SECTION( "KorobovTerm<D,U,1,1,1>" )
    {
        D x_goal[] = {2.8, 2.6, 2.4, 2.2, 2., 1.8, 1.6, 1.4, 1.2};
        for(U s = 0; s < dim; s++)
        {
            y[s] = integrators::transforms::detail::KorobovTerm<D,U,1,1,1>::value(x[s]);
        }
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( y[0] == x_goal[0] );
        }
    };

    SECTION( "KorobovTerm<D,U,2,2,2>" )
    {
        D x_goal[] = {8.56, 7.24, 6.04, 4.96, 4., 3.16, 2.44, 1.84, 1.36};
        for(U s = 0; s < dim; s++)
        {
            y[s] = integrators::transforms::detail::KorobovTerm<D,U,2,2,2>::value(x[s]);
        }
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( y[0] == x_goal[0] );
        }
    };

    SECTION( "KorobovTerm<D,U,3,3,3>" )
    {
        D x_goal[] = {27.28, 20.84, 15.56, 11.32, 8., 5.48, 3.64, 2.36, 1.52};
        for(U s = 0; s < dim; s++)
        {
            y[s] = integrators::transforms::detail::KorobovTerm<D,U,3,3,3>::value(x[s]);
        }
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( y[0] == x_goal[0] );
        }
    };

    SECTION( "KorobovTerm<D,U,4,4,4>" )
    {
        D x_goal[] = {89.092, 61.192, 40.662, 26.032, 16., 9.432, 5.362, 2.992, 1.692};
        for(U s = 0; s < dim; s++)
        {
            y[s] = integrators::transforms::detail::KorobovTerm<D,U,4,4,4>::value(x[s]);
        }
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( y[0] == x_goal[0] );
        }
    };

    SECTION( "KorobovTerm<D,U,8,8,8>" )
    {
        D x_goal[] = {11464.435997200, 5041.919603200, 2046.28034620, 758.88248320, 256.00000000, 79.48875520, 23.7828322, 7.4313472, 2.5811452};
        for(U s = 0; s < dim; s++)
        {
            y[s] = integrators::transforms::detail::KorobovTerm<D,U,8,8,8>::value(x[s]);
        }
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( y[0] == x_goal[0] );
        }
    };

};

TEST_CASE( "Korobov Transform", "[transform]")
{
    using D = double;
    using U = unsigned long long int;

    U dim = 9;

    D x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    D wgt = 1.;

    SECTION( "Korobov<D,U,1>" )
    {
        D x_goal[] = {std::stod("7")/std::stod("250"),std::stod("13")/std::stod("125"),std::stod("27")/std::stod("125"),std::stod("44")/std::stod("125"),std::stod("1")/std::stod("2"),std::stod("81")/std::stod("125"),std::stod("98")/std::stod("125"),std::stod("112")/std::stod("125"),std::stod("243")/std::stod("250")};
        D wgt_goal = std::stod("202491775584")/std::stod("152587890625");

        integrators::transforms::Korobov<D,U,1> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

    SECTION( "Korobov<D,U,2>" )
    {
        D x_goal[] = {std::stod("107")/std::stod("12500"),std::stod("181")/std::stod("3125"),std::stod("4077")/std::stod("25000"),std::stod("992")/std::stod("3125"),std::stod("1")/std::stod("2"),std::stod("2133")/std::stod("3125"),std::stod("20923")/std::stod("25000"),std::stod("2944")/std::stod("3125"),std::stod("12393")/std::stod("12500")};
        D wgt_goal = std::stod("4068679902545286")/std::stod("11920928955078125");

        integrators::transforms::Korobov<D,U,2> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

    SECTION( "Korobov<D,U,3>" )
    {
        D x_goal[] = {std::stod("341")/std::stod("125000"),std::stod("521")/std::stod("15625"),std::stod("31509")/std::stod("250000"),std::stod("4528")/std::stod("15625"),std::stod("1")/std::stod("2"),std::stod("11097")/std::stod("15625"),std::stod("218491")/std::stod("250000"),std::stod("15104")/std::stod("15625"),std::stod("124659")/std::stod("125000")};
        D wgt_goal = std::stod("85814502186767229614757312")/std::stod("1818989403545856475830078125");

        integrators::transforms::Korobov<D,U,3> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

    SECTION( "Korobov<D,U,4>" )
    {
        D x_goal[] = {std::stod("22273")/std::stod("25000000"),std::stod("7649")/std::stod("390625"),std::stod("4940433")/std::stod("50000000"),std::stod("104128")/std::stod("390625"),std::stod("1")/std::stod("2"),std::stod("286497")/std::stod("390625"),std::stod("45059567")/std::stod("50000000"),std::stod("382976")/std::stod("390625"),std::stod("24977727")/std::stod("25000000")};
        D wgt_goal = std::stod("167004977867137272381311782609848204543")/std::stod("35527136788005009293556213378906250000000");

        integrators::transforms::Korobov<D,U,4> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

    SECTION( "Korobov<D,U,8>" )
    {
        D x_goal[] = {std::stod("28661089993")/std::stod("2500000000000000"),std::stod("393899969")/std::stod("152587890625"),std::stod("201384680271273")/std::stod("5000000000000000"),std::stod("30355299328")/std::stod("152587890625"),std::stod("1")/std::stod("2"),std::stod("122232591297")/std::stod("152587890625"),std::stod("4798615319728727")/std::stod("5000000000000000"),std::stod("152193990656")/std::stod("152587890625"),std::stod("2499971338910007")/std::stod("2500000000000000")};
        D wgt_goal = std::stod("131091990195656860194070898236198433121817941509319027782237707310507220432586957239")/std::stod("1262177448353618888658765704452457967477130296174436807632446289062500000000000000000000000");

        integrators::transforms::Korobov<D,U,8> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

};

TEST_CASE( "Tent Transform", "[transform]")
{
    using D = double;
    using U = unsigned long long int;

    U dim = 9;

    D x[]      = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    D wgt = 1.;

    D x_goal[] = {0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2};
    D wgt_goal = 1.;

    SECTION( "Trivial" )
    {
        integrators::transforms::Tent<D> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

    SECTION( "Trivial" )
    {
        integrators::transforms::Tent<D,U> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

};

TEST_CASE( "Trivial Transform", "[transform]")
{
    using D = double;
    using U = unsigned long long int;

    U dim = 9;

    D x[]      = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    D wgt = 1.;

    D x_goal[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    D wgt_goal = 1.;

    SECTION( "Trivial" )
    {
        integrators::transforms::Trivial<D> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

    SECTION( "Trivial" )
    {
        integrators::transforms::Trivial<D,U> transform;
        transform(x,wgt,dim);
        for(U s = 0; s < dim; s++)
        {
            REQUIRE( x[s] == Approx(x_goal[s]) );
        }
        REQUIRE( wgt == Approx(wgt_goal) );
    };

};
