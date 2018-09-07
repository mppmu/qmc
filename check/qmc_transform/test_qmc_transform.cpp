#include "catch.hpp"
#include "qmc.hpp"

#include <string> // stod

#ifdef __CUDACC__
#include <thrust/complex.h>
using thrust::complex;
#define HOSTDEVICE __host__ __device__
#else
using std::complex;
#include <complex>
#define HOSTDEVICE
#endif

TEST_CASE( "ipow", "[ipow]")
{

    SECTION( "Integer Raised to Unsigned Integer" )
    {
        using D = int;
        using U = unsigned long long int;

        REQUIRE(integrators::transforms::detail::IPow<D,U,0>::value(1) == 1 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,0>::value(3) == 1 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,1>::value(2) == 2 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,1>::value(-2) == -2 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,3>::value(2) == 8 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,3>::value(-2) == -8 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,10>::value(3) == 59049 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,10>::value(-3) == 59049 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,9>::value(4) == 262144 );
        REQUIRE(integrators::transforms::detail::IPow<D,U,9>::value(-4) == -262144 );

    };

    SECTION( "Double Raised to Unsigned Integer" )
    {
        using D = double;
        using U = unsigned long long int;

        REQUIRE(integrators::transforms::detail::IPow<D,U,0>::value(1.) == Approx(1.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,0>::value(3.) == Approx(1.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,1>::value(2.) == Approx(2.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,1>::value(-2.) == Approx(-2.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,3>::value(2.) == Approx(8.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,3>::value(-2.) == Approx(-8.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,10>::value(3.) == Approx(59049.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,10>::value(-3.) == Approx(59049.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,9>::value(4.) == Approx(262144.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,9>::value(-4.) == Approx(-262144.) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,101>::value(-4.) == Approx(-6.42775217703596110e60) );
        REQUIRE(integrators::transforms::detail::IPow<D,U,40>::value(7e-6) == Approx(6.36680576090902799e-207) );
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

        //r0=0,r1=0  x=x
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,0,0>::value() ==1);
        //r0=0,r1=1  x=(2 - x)*x
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,0,1>::value() ==2);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,0,1>::value() ==-1);
        //r0=0,r1=2  x=x*(3 + (-3 + x)*x)
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,0,2>::value() ==3);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,0,2>::value() ==-3);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,0,2>::value() ==1);
        //r0=0,r1=3  x=x*(4 + x*(-6 + (4 - x)*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,0,3>::value() ==4);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,0,3>::value() ==-6);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,0,3>::value() ==4);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,0,3>::value() ==-1);
        //r0=0,r1=4  x=x*(5 + x*(-10 + x*(10 + (-5 + x)*x)))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,0,4>::value() ==5);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,0,4>::value() ==-10);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,0,4>::value() ==10);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,0,4>::value() ==-5);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,0,4>::value() ==1);
        //r0=0,r1=5  x=x*(6 + x*(-15 + x*(20 + x*(-15 + (6 - x)*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,0,5>::value() ==6);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,0,5>::value() ==-15);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,0,5>::value() ==20);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,0,5>::value() ==-15);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,0,5>::value() ==6);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,5,0,5>::value() ==-1);
        //r0=1,r1=0  x=x^2
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,0>::value() ==1);
        //r0=1,r1=1  x=(3 - 2*x)*x^2
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,1>::value() ==3);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,1,1>::value() ==-2);
        //r0=1,r1=2  x=x^2*(6 + x*(-8 + 3*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,2>::value() ==6);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,1,2>::value() ==-8);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,1,2>::value() ==3);
        //r0=1,r1=3  x=x^2*(10 + x*(-20 + (15 - 4*x)*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,3>::value() ==10);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,1,3>::value() ==-20);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,1,3>::value() ==15);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,1,3>::value() ==-4);
        //r0=1,r1=4  x=x^2*(15 + x*(-40 + x*(45 + x*(-24 + 5*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,4>::value() ==15);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,1,4>::value() ==-40);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,1,4>::value() ==45);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,1,4>::value() ==-24);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,1,4>::value() ==5);
        //r0=1,r1=5  x=x^2*(21 + x*(-70 + x*(105 + x*(-84 + (35 - 6*x)*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,1,5>::value() ==21);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,1,5>::value() ==-70);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,1,5>::value() ==105);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,1,5>::value() ==-84);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,1,5>::value() ==35);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,5,1,5>::value() ==-6);
        //r0=2,r1=0  x=x^3
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,0>::value() ==1);
        //r0=2,r1=1  x=(4 - 3*x)*x^3
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,1>::value() ==4);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,2,1>::value() ==-3);
        //r0=2,r1=2  x=x^3*(10 + x*(-15 + 6*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,2>::value() ==10);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,2,2>::value() ==-15);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,2,2>::value() ==6);
        //r0=2,r1=3  x=x^3*(20 + x*(-45 + (36 - 10*x)*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,3>::value() ==20);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,2,3>::value() ==-45);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,2,3>::value() ==36);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,2,3>::value() ==-10);
        //r0=2,r1=4  x=x^3*(35 + x*(-105 + x*(126 + x*(-70 + 15*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,4>::value() ==35);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,2,4>::value() ==-105);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,2,4>::value() ==126);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,2,4>::value() ==-70);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,2,4>::value() ==15);
        //r0=2,r1=5  x=x^3*(56 + x*(-210 + x*(336 + x*(-280 + (120 - 21*x)*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,2,5>::value() ==56);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,2,5>::value() ==-210);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,2,5>::value() ==336);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,2,5>::value() ==-280);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,2,5>::value() ==120);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,5,2,5>::value() ==-21);
        //r0=3,r1=0  x=x^4
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,0>::value() ==1);
        //r0=3,r1=1  x=(5 - 4*x)*x^4
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,1>::value() ==5);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,3,1>::value() ==-4);
        //r0=3,r1=2  x=x^4*(15 + x*(-24 + 10*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,2>::value() ==15);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,3,2>::value() ==-24);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,3,2>::value() ==10);
        //r0=3,r1=3  x=x^4*(35 + x*(-84 + (70 - 20*x)*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,3>::value() ==35);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,3,3>::value() ==-84);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,3,3>::value() ==70);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,3,3>::value() ==-20);
        //r0=3,r1=4  x=x^4*(70 + x*(-224 + x*(280 + x*(-160 + 35*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,4>::value() ==70);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,3,4>::value() ==-224);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,3,4>::value() ==280);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,3,4>::value() ==-160);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,3,4>::value() ==35);
        //r0=3,r1=5  x=x^4*(126 + x*(-504 + x*(840 + x*(-720 + (315 - 56*x)*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,3,5>::value() ==126);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,3,5>::value() ==-504);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,3,5>::value() ==840);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,3,5>::value() ==-720);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,3,5>::value() ==315);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,5,3,5>::value() ==-56);
        //r0=4,r1=0  x=x^5
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,0>::value() ==1);
        //r0=4,r1=1  x=(6 - 5*x)*x^5
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,1>::value() ==6);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,4,1>::value() ==-5);
        //r0=4,r1=2  x=x^5*(21 + x*(-35 + 15*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,2>::value() ==21);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,4,2>::value() ==-35);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,4,2>::value() ==15);
        //r0=4,r1=3  x=x^5*(56 + x*(-140 + (120 - 35*x)*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,3>::value() ==56);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,4,3>::value() ==-140);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,4,3>::value() ==120);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,4,3>::value() ==-35);
        //r0=4,r1=4  x=x^5*(126 + x*(-420 + x*(540 + x*(-315 + 70*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,4>::value() ==126);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,4,4>::value() ==-420);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,4,4>::value() ==540);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,4,4>::value() ==-315);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,4,4>::value() ==70);
        //r0=4,r1=5  x=x^5*(252 + x*(-1050 + x*(1800 + x*(-1575 + (700 - 126*x)*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,4,5>::value() ==252);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,4,5>::value() ==-1050);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,4,5>::value() ==1800);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,4,5>::value() ==-1575);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,4,5>::value() ==700);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,5,4,5>::value() ==-126);
        //r0=5,r1=0  x=x^6
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,5,0>::value() ==1);
        //r0=5,r1=1  x=(7 - 6*x)*x^6
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,5,1>::value() ==7);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,5,1>::value() ==-6);
        //r0=5,r1=2  x=x^6*(28 + x*(-48 + 21*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,5,2>::value() ==28);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,5,2>::value() ==-48);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,5,2>::value() ==21);
        //r0=5,r1=3  x=x^6*(84 + x*(-216 + (189 - 56*x)*x))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,5,3>::value() ==84);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,5,3>::value() ==-216);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,5,3>::value() ==189);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,5,3>::value() ==-56);
        //r0=5,r1=4  x=x^6*(210 + x*(-720 + x*(945 + x*(-560 + 126*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,5,4>::value() ==210);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,5,4>::value() ==-720);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,5,4>::value() ==945);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,5,4>::value() ==-560);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,5,4>::value() ==126);
        //r0=5,r1=5  x=x^6*(462 + x*(-1980 + x*(3465 + x*(-3080 + (1386 - 252*x)*x))))
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,0,5,5>::value() ==462);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,1,5,5>::value() ==-1980);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,2,5,5>::value() ==3465);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,3,5,5>::value() ==-3080);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,4,5,5>::value() ==1386);
        REQUIRE(integrators::transforms::detail::KorobovCoefficient<D,U,5,5,5>::value() ==-252);

     };

    SECTION( "Double Coefficients" )
    {
        using D = double;
        using U = unsigned long long int;

        double korobov_coefficient; // workaround catch issue

        //r0=0,r1=0  x=x
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,0,0>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=0,r1=1  x=(2 - x)*x
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,0,1>::value();
        REQUIRE(korobov_coefficient == Approx(2.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,0,1>::value();
        REQUIRE(korobov_coefficient == Approx(-1.) );
        //r0=0,r1=2  x=x*(3 + (-3 + x)*x)
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,0,2>::value();
        REQUIRE(korobov_coefficient == Approx(3.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,0,2>::value();
        REQUIRE(korobov_coefficient == Approx(-3.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,0,2>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=0,r1=3  x=x*(4 + x*(-6 + (4 - x)*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,0,3>::value();
        REQUIRE(korobov_coefficient == Approx(4.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,0,3>::value();
        REQUIRE(korobov_coefficient == Approx(-6.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,0,3>::value();
        REQUIRE(korobov_coefficient == Approx(4.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,0,3>::value();
        REQUIRE(korobov_coefficient == Approx(-1.) );
        //r0=0,r1=4  x=x*(5 + x*(-10 + x*(10 + (-5 + x)*x)))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,0,4>::value();
        REQUIRE(korobov_coefficient == Approx(5.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,0,4>::value();
        REQUIRE(korobov_coefficient == Approx(-10.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,0,4>::value();
        REQUIRE(korobov_coefficient == Approx(10.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,0,4>::value();
        REQUIRE(korobov_coefficient == Approx(-5.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,0,4>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=0,r1=5  x=x*(6 + x*(-15 + x*(20 + x*(-15 + (6 - x)*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,0,5>::value();
        REQUIRE(korobov_coefficient == Approx(6.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,0,5>::value();
        REQUIRE(korobov_coefficient == Approx(-15.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,0,5>::value();
        REQUIRE(korobov_coefficient == Approx(20.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,0,5>::value();
        REQUIRE(korobov_coefficient == Approx(-15.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,0,5>::value();
        REQUIRE(korobov_coefficient == Approx(6.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,5,0,5>::value();
        REQUIRE(korobov_coefficient == Approx(-1.) );
        //r0=1,r1=0  x=x^2
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,1,0>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=1,r1=1  x=(3 - 2*x)*x^2
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,1,1>::value();
        REQUIRE(korobov_coefficient == Approx(3.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,1,1>::value();
        REQUIRE(korobov_coefficient == Approx(-2.) );
        //r0=1,r1=2  x=x^2*(6 + x*(-8 + 3*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,1,2>::value();
        REQUIRE(korobov_coefficient == Approx(6.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,1,2>::value();
        REQUIRE(korobov_coefficient == Approx(-8.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,1,2>::value();
        REQUIRE(korobov_coefficient == Approx(3.) );
        //r0=1,r1=3  x=x^2*(10 + x*(-20 + (15 - 4*x)*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,1,3>::value();
        REQUIRE(korobov_coefficient == Approx(10.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,1,3>::value();
        REQUIRE(korobov_coefficient == Approx(-20.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,1,3>::value();
        REQUIRE(korobov_coefficient == Approx(15.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,1,3>::value();
        REQUIRE(korobov_coefficient == Approx(-4.) );
        //r0=1,r1=4  x=x^2*(15 + x*(-40 + x*(45 + x*(-24 + 5*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,1,4>::value();
        REQUIRE(korobov_coefficient == Approx(15.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,1,4>::value();
        REQUIRE(korobov_coefficient == Approx(-40.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,1,4>::value();
        REQUIRE(korobov_coefficient == Approx(45.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,1,4>::value();
        REQUIRE(korobov_coefficient == Approx(-24.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,1,4>::value();
        REQUIRE(korobov_coefficient == Approx(5.) );
        //r0=1,r1=5  x=x^2*(21 + x*(-70 + x*(105 + x*(-84 + (35 - 6*x)*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,1,5>::value();
        REQUIRE(korobov_coefficient == Approx(21.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,1,5>::value();
        REQUIRE(korobov_coefficient == Approx(-70.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,1,5>::value();
        REQUIRE(korobov_coefficient == Approx(105.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,1,5>::value();
        REQUIRE(korobov_coefficient == Approx(-84.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,1,5>::value();
        REQUIRE(korobov_coefficient == Approx(35.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,5,1,5>::value();
        REQUIRE(korobov_coefficient == Approx(-6.) );
        //r0=2,r1=0  x=x^3
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,2,0>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=2,r1=1  x=(4 - 3*x)*x^3
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,2,1>::value();
        REQUIRE(korobov_coefficient == Approx(4.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,2,1>::value();
        REQUIRE(korobov_coefficient == Approx(-3.) );
        //r0=2,r1=2  x=x^3*(10 + x*(-15 + 6*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,2,2>::value();
        REQUIRE(korobov_coefficient == Approx(10.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,2,2>::value();
        REQUIRE(korobov_coefficient == Approx(-15.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,2,2>::value();
        REQUIRE(korobov_coefficient == Approx(6.) );
        //r0=2,r1=3  x=x^3*(20 + x*(-45 + (36 - 10*x)*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,2,3>::value();
        REQUIRE(korobov_coefficient == Approx(20.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,2,3>::value();
        REQUIRE(korobov_coefficient == Approx(-45.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,2,3>::value();
        REQUIRE(korobov_coefficient == Approx(36.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,2,3>::value();
        REQUIRE(korobov_coefficient == Approx(-10.) );
        //r0=2,r1=4  x=x^3*(35 + x*(-105 + x*(126 + x*(-70 + 15*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,2,4>::value();
        REQUIRE(korobov_coefficient == Approx(35.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,2,4>::value();
        REQUIRE(korobov_coefficient == Approx(-105.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,2,4>::value();
        REQUIRE(korobov_coefficient == Approx(126.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,2,4>::value();
        REQUIRE(korobov_coefficient == Approx(-70.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,2,4>::value();
        REQUIRE(korobov_coefficient == Approx(15.) );
        //r0=2,r1=5  x=x^3*(56 + x*(-210 + x*(336 + x*(-280 + (120 - 21*x)*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,2,5>::value();
        REQUIRE(korobov_coefficient == Approx(56.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,2,5>::value();
        REQUIRE(korobov_coefficient == Approx(-210.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,2,5>::value();
        REQUIRE(korobov_coefficient == Approx(336.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,2,5>::value();
        REQUIRE(korobov_coefficient == Approx(-280.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,2,5>::value();
        REQUIRE(korobov_coefficient == Approx(120.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,5,2,5>::value();
        REQUIRE(korobov_coefficient == Approx(-21.) );
        //r0=3,r1=0  x=x^4
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,3,0>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=3,r1=1  x=(5 - 4*x)*x^4
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,3,1>::value();
        REQUIRE(korobov_coefficient == Approx(5.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,3,1>::value();
        REQUIRE(korobov_coefficient == Approx(-4.) );
        //r0=3,r1=2  x=x^4*(15 + x*(-24 + 10*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,3,2>::value();
        REQUIRE(korobov_coefficient == Approx(15.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,3,2>::value();
        REQUIRE(korobov_coefficient == Approx(-24.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,3,2>::value();
        REQUIRE(korobov_coefficient == Approx(10.) );
        //r0=3,r1=3  x=x^4*(35 + x*(-84 + (70 - 20*x)*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,3,3>::value();
        REQUIRE(korobov_coefficient == Approx(35.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,3,3>::value();
        REQUIRE(korobov_coefficient == Approx(-84.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,3,3>::value();
        REQUIRE(korobov_coefficient == Approx(70.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,3,3>::value();
        REQUIRE(korobov_coefficient == Approx(-20.) );
        //r0=3,r1=4  x=x^4*(70 + x*(-224 + x*(280 + x*(-160 + 35*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,3,4>::value();
        REQUIRE(korobov_coefficient == Approx(70.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,3,4>::value();
        REQUIRE(korobov_coefficient == Approx(-224.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,3,4>::value();
        REQUIRE(korobov_coefficient == Approx(280.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,3,4>::value();
        REQUIRE(korobov_coefficient == Approx(-160.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,3,4>::value();
        REQUIRE(korobov_coefficient == Approx(35.) );
        //r0=3,r1=5  x=x^4*(126 + x*(-504 + x*(840 + x*(-720 + (315 - 56*x)*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,3,5>::value();
        REQUIRE(korobov_coefficient == Approx(126.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,3,5>::value();
        REQUIRE(korobov_coefficient == Approx(-504.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,3,5>::value();
        REQUIRE(korobov_coefficient == Approx(840.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,3,5>::value();
        REQUIRE(korobov_coefficient == Approx(-720.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,3,5>::value();
        REQUIRE(korobov_coefficient == Approx(315.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,5,3,5>::value();
        REQUIRE(korobov_coefficient == Approx(-56.) );
        //r0=4,r1=0  x=x^5
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,4,0>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=4,r1=1  x=(6 - 5*x)*x^5
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,4,1>::value();
        REQUIRE(korobov_coefficient == Approx(6.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,4,1>::value();
        REQUIRE(korobov_coefficient == Approx(-5.) );
        //r0=4,r1=2  x=x^5*(21 + x*(-35 + 15*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,4,2>::value();
        REQUIRE(korobov_coefficient == Approx(21.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,4,2>::value();
        REQUIRE(korobov_coefficient == Approx(-35.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,4,2>::value();
        REQUIRE(korobov_coefficient == Approx(15.) );
        //r0=4,r1=3  x=x^5*(56 + x*(-140 + (120 - 35*x)*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,4,3>::value();
        REQUIRE(korobov_coefficient == Approx(56.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,4,3>::value();
        REQUIRE(korobov_coefficient == Approx(-140.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,4,3>::value();
        REQUIRE(korobov_coefficient == Approx(120.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,4,3>::value();
        REQUIRE(korobov_coefficient == Approx(-35.) );
        //r0=4,r1=4  x=x^5*(126 + x*(-420 + x*(540 + x*(-315 + 70*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,4,4>::value();
        REQUIRE(korobov_coefficient == Approx(126.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,4,4>::value();
        REQUIRE(korobov_coefficient == Approx(-420.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,4,4>::value();
        REQUIRE(korobov_coefficient == Approx(540.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,4,4>::value();
        REQUIRE(korobov_coefficient == Approx(-315.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,4,4>::value();
        REQUIRE(korobov_coefficient == Approx(70.) );
        //r0=4,r1=5  x=x^5*(252 + x*(-1050 + x*(1800 + x*(-1575 + (700 - 126*x)*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,4,5>::value();
        REQUIRE(korobov_coefficient == Approx(252.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,4,5>::value();
        REQUIRE(korobov_coefficient == Approx(-1050.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,4,5>::value();
        REQUIRE(korobov_coefficient == Approx(1800.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,4,5>::value();
        REQUIRE(korobov_coefficient == Approx(-1575.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,4,5>::value();
        REQUIRE(korobov_coefficient == Approx(700.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,5,4,5>::value();
        REQUIRE(korobov_coefficient == Approx(-126.) );
        //r0=5,r1=0  x=x^6
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,5,0>::value();
        REQUIRE(korobov_coefficient == Approx(1.) );
        //r0=5,r1=1  x=(7 - 6*x)*x^6
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,5,1>::value();
        REQUIRE(korobov_coefficient == Approx(7.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,5,1>::value();
        REQUIRE(korobov_coefficient == Approx(-6.) );
        //r0=5,r1=2  x=x^6*(28 + x*(-48 + 21*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,5,2>::value();
        REQUIRE(korobov_coefficient == Approx(28.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,5,2>::value();
        REQUIRE(korobov_coefficient == Approx(-48.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,5,2>::value();
        REQUIRE(korobov_coefficient == Approx(21.) );
        //r0=5,r1=3  x=x^6*(84 + x*(-216 + (189 - 56*x)*x))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,5,3>::value();
        REQUIRE(korobov_coefficient == Approx(84.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,5,3>::value();
        REQUIRE(korobov_coefficient == Approx(-216.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,5,3>::value();
        REQUIRE(korobov_coefficient == Approx(189.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,5,3>::value();
        REQUIRE(korobov_coefficient == Approx(-56.) );
        //r0=5,r1=4  x=x^6*(210 + x*(-720 + x*(945 + x*(-560 + 126*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,5,4>::value();
        REQUIRE(korobov_coefficient == Approx(210.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,5,4>::value();
        REQUIRE(korobov_coefficient == Approx(-720.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,5,4>::value();
        REQUIRE(korobov_coefficient == Approx(945.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,5,4>::value();
        REQUIRE(korobov_coefficient == Approx(-560.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,5,4>::value();
        REQUIRE(korobov_coefficient == Approx(126.) );
        //r0=5,r1=5  x=x^6*(462 + x*(-1980 + x*(3465 + x*(-3080 + (1386 - 252*x)*x))))
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,0,5,5>::value();
        REQUIRE(korobov_coefficient == Approx(462.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,1,5,5>::value();
        REQUIRE(korobov_coefficient == Approx(-1980.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,2,5,5>::value();
        REQUIRE(korobov_coefficient == Approx(3465.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,3,5,5>::value();
        REQUIRE(korobov_coefficient == Approx(-3080.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,4,5,5>::value();
        REQUIRE(korobov_coefficient == Approx(1386.) );
        korobov_coefficient=integrators::transforms::detail::KorobovCoefficient<D,U,5,5,5>::value();
        REQUIRE(korobov_coefficient == Approx(-252.) );

    };

};

TEST_CASE( "KorobovTerm", "[transform]")
{
    using D = double;
    using U = unsigned long long int;

    const U dim = 9;
    const D x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    D y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    SECTION( "KorobovTerm<D,U,1,1,1>" )
    {
        const D x_goal[] = {2.8, 2.6, 2.4, 2.2, 2., 1.8, 1.6, 1.4, 1.2};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,1,1>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,1,2>" )
    {
        const D x_goal[] = {-7.7, -7.4, -7.1, -6.8, -6.5, -6.2, -5.9, -5.6, -5.3};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,1,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,1,2>" )
    {
        const D x_goal[] = {5.23, 4.52, 3.87, 3.28, 2.75, 2.28, 1.87, 1.52, 1.23};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,1,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,1,3>" )
    {
        const D x_goal[] = {14.6, 14.2, 13.8, 13.4, 13., 12.6, 12.2, 11.8, 11.4};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,1,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,1,3>" )
    {
        const D x_goal[] = {-18.54, -17.16, -15.86, -14.64, -13.5, -12.44, -11.46, -10.56, -9.74};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,1,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,1,3>" )
    {
        const D x_goal[] = {8.146, 6.568, 5.242, 4.144, 3.25, 2.536, 1.978, 1.552, 1.234};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,1,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,1,4>" )
    {
        const D x_goal[] = {-23.5, -23., -22.5, -22., -21.5, -21., -20.5, -20., -19.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,1,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,1,4>" )
    {
        const D x_goal[] = {42.65, 40.4, 38.25, 36.2, 34.25, 32.4, 30.65, 29., 27.45};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,1,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,1,4>" )
    {
        const D x_goal[] = {-35.735, -31.92, -28.525, -25.52, -22.875, -20.56, -18.545, -16.8, -15.295};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,1,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,4,1,4>" )
    {
        const D x_goal[] = {11.4265, 8.616, 6.4425, 4.792, 3.5625, 2.664, 2.0185, 1.56, 1.2345};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,4,1,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,2,1>" )
    {
        const D x_goal[] = {3.7, 3.4, 3.1, 2.8, 2.5, 2.2, 1.9, 1.6, 1.3};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,2,1>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,2,2>" )
    {
        const D x_goal[] = {-14.4, -13.8, -13.2, -12.6, -12., -11.4, -10.8, -10.2, -9.6};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,2,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,2,2>" )
    {
        const D x_goal[] = {8.56, 7.24, 6.04, 4.96, 4., 3.16, 2.44, 1.84, 1.36};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,2,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,2,3>" )
    {
        const D x_goal[] = {35., 34., 33., 32., 31., 30., 29., 28., 27.};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,2,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,2,3>" )
    {
        const D x_goal[] = {-41.5, -38.2, -35.1, -32.2, -29.5, -27., -24.7, -22.6, -20.7};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,2,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,2,3>" )
    {
        const D x_goal[] = {15.85, 12.36, 9.47, 7.12, 5.25, 3.8, 2.71, 1.92, 1.37};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,2,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,2,4>" )
    {
        const D x_goal[] = {-68.5, -67., -65.5, -64., -62.5, -61., -59.5, -58., -56.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,2,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,2,4>" )
    {
        const D x_goal[] = {119.15, 112.6, 106.35, 100.4, 94.75, 89.4, 84.35, 79.6, 75.15};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,2,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,2,4>" )
    {
        const D x_goal[] = {-93.085, -82.48, -73.095, -64.84, -57.625, -51.36, -45.955, -41.32, -37.365};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,2,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,4,2,4>" )
    {
        const D x_goal[] = {25.6915, 18.504, 13.0715, 9.064, 6.1875, 4.184, 2.8315, 1.944, 1.3715};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,4,2,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,3,1>" )
    {
        const D x_goal[] = {4.6, 4.2, 3.8, 3.4, 3., 2.6, 2.2, 1.8, 1.4};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,3,1>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,3,2>" )
    {
        const D x_goal[] = {-23., -22., -21., -20., -19., -18., -17., -16., -15.};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,3,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,3,2>" )
    {
        const D x_goal[] = {12.7, 10.6, 8.7, 7., 5.5, 4.2, 3.1, 2.2, 1.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,3,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,3,3>" )
    {
        const D x_goal[] = {68., 66., 64., 62., 60., 58., 56., 54., 52.};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,3,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,3,3>" )
    {
        const D x_goal[] = {-77.2, -70.8, -64.8, -59.2, -54., -49.2, -44.8, -40.8, -37.2};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,3,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,3,3>" )
    {
        const D x_goal[] = {27.28, 20.84, 15.56, 11.32, 8., 5.48, 3.64, 2.36, 1.52};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,3,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,3,4>" )
    {
        const D x_goal[] = {-156.5, -153., -149.5, -146., -142.5, -139., -135.5, -132., -128.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,3,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,3,4>" )
    {
        const D x_goal[] = {264.35, 249.4, 235.15, 221.6, 208.75, 196.6, 185.15, 174.4, 164.35};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,3,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,3,4>" )
    {
        const D x_goal[] = {-197.565, -174.12, -153.455, -135.36, -119.625, -106.04, -94.395, -84.48, -76.085};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,3,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,4,3,4>" )
    {
        const D x_goal[] = {50.2435, 35.176, 23.9635, 15.856, 10.1875, 6.376, 3.9235, 2.416, 1.5235};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,4,3,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,4,1>" )
    {
        const D x_goal[] = {5.5, 5., 4.5, 4., 3.5, 3., 2.5, 2., 1.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,4,1>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,4,2>" )
    {
        const D x_goal[] = {-33.5, -32., -30.5, -29., -27.5, -26., -24.5, -23., -21.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,4,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,4,2>" )
    {
        const D x_goal[] = {17.65, 14.6, 11.85, 9.4, 7.25, 5.4, 3.85, 2.6, 1.65};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,4,2>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,4,3>" )
    {
        const D x_goal[] = {116.5, 113., 109.5, 106., 102.5, 99., 95.5, 92., 88.5};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,4,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,4,3>" )
    {
        const D x_goal[] = {-128.35, -117.4, -107.15, -97.6, -88.75, -80.6, -73.15, -66.4, -60.35};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,4,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,4,3>" )
    {
        const D x_goal[] = {43.165, 32.52, 23.855, 16.96, 11.625, 7.64, 4.795, 2.88, 1.685};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,4,3>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,1,4,4>" )
    {
        const D x_goal[] = {-308., -301., -294., -287., -280., -273., -266., -259., -252.};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,1,4,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,2,4,4>" )
    {
        const D x_goal[] = {509.2, 479.8, 451.8, 425.2, 400., 376.2, 353.8, 332.8, 313.2};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,2,4,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,3,4,4>" )
    {
        const D x_goal[] = {-369.08, -324.04, -284.46, -249.92, -220., -194.28, -172.34, -153.76, -138.12};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,3,4,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

    SECTION( "KorobovTerm<D,U,4,4,4>" )
    {
        const D x_goal[] = {89.092, 61.192, 40.662, 26.032, 16., 9.432, 5.362, 2.992, 1.692};
        for(U s=0;s<dim;s++)
        {
            y[s]=integrators::transforms::detail::KorobovTerm<D,U,4,4,4>::value(x[s]);
        }
        for(U s=0;s<dim;s++)
        {
            REQUIRE( y[s] == Approx(x_goal[s]) );
        }
    };

};

struct trivial_functor_t {
    const unsigned long long int dim = 1;
    HOSTDEVICE double operator()(double* x) { return x[0]; }
} trivial_functor;

TEST_CASE( "Korobov Transform", "[transform]")
{
    using D = double;
    using U = unsigned long long int;

    const U num_tests = 9;
    D x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    SECTION( "Korobov<D,U,0,0>" )
    {
        const D x_goal[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        integrators::transforms::Korobov<trivial_functor_t,D,U,0,0> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,0,1>" )
    {
        const D x_goal[] = {0.342, 0.576, 0.714, 0.768, 0.75, 0.672, 0.546, 0.384, 0.198};
        integrators::transforms::Korobov<trivial_functor_t,D,U,0,1> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,0,2>" )
    {
        const D x_goal[] = {0.65853, 0.93696, 0.96579, 0.84672, 0.65625, 0.44928, 0.26271, 0.11904, 0.02997};
        integrators::transforms::Korobov<trivial_functor_t,D,U,0,2> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,0,3>" )
    {
        const D x_goal[] = {1.0028124, 1.2091392, 1.0425828, 0.7520256, 0.46875, 0.2494464, 0.1071252, 0.0319488, 0.0039996};
        integrators::transforms::Korobov<trivial_functor_t,D,U,0,3> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,0,4>" )
    {
        const D x_goal[] = {1.343397555, 1.37691136, 0.998731965, 0.59761152, 0.302734375, 0.12668928, 0.040401585, 0.00799744, 0.000499995};
        integrators::transforms::Korobov<trivial_functor_t,D,U,0,4> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,1,0>" )
    {
        const D x_goal[] = {0.002, 0.016, 0.054, 0.128, 0.25, 0.432, 0.686, 1.024, 1.458};
        integrators::transforms::Korobov<trivial_functor_t,D,U,1,0> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,1,1>" )
    {
        const D x_goal[] = {0.01512, 0.09984, 0.27216, 0.50688, 0.75, 0.93312, 0.98784, 0.86016, 0.52488};
        integrators::transforms::Korobov<trivial_functor_t,D,U,1,1> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,1,2>" )
    {
        const D x_goal[] = {0.0508356, 0.2777088, 0.6144012, 0.9068544, 1.03125, 0.9455616, 0.6927228, 0.3735552, 0.1076004};
        integrators::transforms::Korobov<trivial_functor_t,D,U,1,2> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,1,3>" )
    {
        const D x_goal[] = {0.11876868, 0.53805056, 0.97092324, 1.14573312, 1.015625, 0.70115328, 0.36636516, 0.12713984, 0.01799172};
        integrators::transforms::Korobov<trivial_functor_t,D,U,1,3> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,1,4>" )
    {
        const D x_goal[] = {0.2249077995, 0.846987264, 1.2529438425, 1.192402944, 0.8349609375, 0.441925632, 0.1682399565, 0.03833856, 0.0026998515};
        integrators::transforms::Korobov<trivial_functor_t,D,U,1,4> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,2,0>" )
    {
        const D x_goal[] = {0.00003, 0.00096, 0.00729, 0.03072, 0.09375, 0.23328, 0.50421, 0.98304, 1.77147};
        integrators::transforms::Korobov<trivial_functor_t,D,U,2,0> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,2,1>" )
    {
        const D x_goal[] = {0.0003996, 0.0104448, 0.0632772, 0.2064384, 0.46875, 0.8211456, 1.1495988, 1.2582912, 0.9211644};
        integrators::transforms::Korobov<trivial_functor_t,D,U,2,1> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,2,2>" )
    {
        const D x_goal[] = {0.00208008, 0.04448256, 0.21575484, 0.54853632, 0.9375, 1.17946368, 1.10724516, 0.72351744, 0.24091992};
        integrators::transforms::Korobov<trivial_functor_t,D,U,2,2> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,2,3>" )
    {
        const D x_goal[] = {0.00693279, 0.121503744, 0.473589018, 0.944898048, 1.23046875, 1.13467392, 0.737860914, 0.301989888, 0.048538278};
        integrators::transforms::Korobov<trivial_functor_t,D,U,2,3> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,2,4>" )
    {
        const D x_goal[] = {0.0176990028075, 0.25466241024, 0.8007798933225, 1.26303141888, 1.2689208984375, 0.87453499392, 0.4047446193525, 0.10701766656, 0.0085034988675};
        integrators::transforms::Korobov<trivial_functor_t,D,U,2,4> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,3,0>" )
    {
        const D x_goal[] = {4.e-7, 0.0000512, 0.0008748, 0.0065536, 0.03125, 0.1119744, 0.3294172, 0.8388608, 1.9131876};
        integrators::transforms::Korobov<trivial_functor_t,D,U,3,0> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,3,1>" )
    {
        const D x_goal[] = {8.28e-6, 0.00086016, 0.01163484, 0.06684672, 0.234375, 0.58226688, 1.08707676, 1.50994944, 1.33923132};
        integrators::transforms::Korobov<trivial_functor_t,D,U,3,1> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,3,2>" )
    {
        const D x_goal[] = {0.000061722, 0.005210112, 0.055939086, 0.24772608, 0.64453125, 1.128701952, 1.378610982, 1.107296256, 0.43046721};
        integrators::transforms::Korobov<trivial_functor_t,D,U,3,2> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,3,3>" )
    {
        const D x_goal[] = {0.00027841968, 0.01912078336, 0.16341071544, 0.56085184512, 1.09375, 1.37450815488, 1.13312928456, 0.55431921664, 0.10178158032};
        integrators::transforms::Korobov<trivial_functor_t,D,U,3,3> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,3,4>" )
    {
        const D x_goal[] = {0.0009230132898, 0.0516385931264, 0.3523296755286, 0.9427067338752, 1.392822265625, 1.2793962037248, 0.7328295738414, 0.2269890215936, 0.0204031891602};
        integrators::transforms::Korobov<trivial_functor_t,D,U,3,4> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,4,0>" )
    {
        const D x_goal[] = {5.e-9, 2.56e-6, 0.000098415, 0.00131072, 0.009765625, 0.05038848, 0.201768035, 0.67108864, 1.937102445};
        integrators::transforms::Korobov<trivial_functor_t,D,U,4,0> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,4,1>" )
    {
        const D x_goal[] = {1.485e-7, 0.00006144, 0.0018600435, 0.018874368, 0.1025390625, 0.362797056, 0.9079561575, 1.610612736, 1.7433922005};
        integrators::transforms::Korobov<trivial_functor_t,D,U,4,1> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,4,2>" )
    {
        const D x_goal[] = {1.5011325e-6, 0.00050233344, 0.0120003806475, 0.09314500608, 0.3717041015625, 0.91424858112, 1.4681651066775, 1.46565758976, 0.6712059971925};
        integrators::transforms::Korobov<trivial_functor_t,D,U,4,2> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,4,3>" )
    {
        const D x_goal[] = {8.8108398e-6, 0.0023869784064, 0.0450944261586, 0.2688917962752, 0.794677734375, 1.3797252661248, 1.4628263244714, 0.8658654068736, 0.1827849867102};
        integrators::transforms::Korobov<trivial_functor_t,D,U,4,3> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

    SECTION( "Korobov<D,U,4,4>" )
    {
        const D x_goal[] = {0.000036825554556, 0.008084722286592, 0.121063364134398, 0.557176779177984, 1.23046875, 1.533012020822016, 1.104166935865602, 0.404792077713408, 0.041297474445444};
        integrators::transforms::Korobov<trivial_functor_t,D,U,4,4> korobov_transform(trivial_functor);
        for(U i=0;i<num_tests;i++)
            REQUIRE( korobov_transform(&x[i]) == Approx(x_goal[i]) );
    };

};

//TEST_CASE( "Baker Transform", "[transform]")
//{
//    using D = double;
//    using U = unsigned long long int;
//
//    U dim = 9;
//
//    D x[]      = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
//    D wgt = 1.;
//
//    D x_goal[] = {0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2};
//    D wgt_goal = 1.;
//
//    SECTION( "Trivial" )
//    {
//        integrators::transforms::Baker<D> transform;
//        transform(x,wgt,dim);
//        for(U s = 0; s < dim; s++)
//        {
//            REQUIRE( x[s] == Approx(x_goal[s]) );
//        }
//        REQUIRE( wgt == Approx(wgt_goal) );
//    };
//
//    SECTION( "Trivial" )
//    {
//        integrators::transforms::Baker<D,U> transform;
//        transform(x,wgt,dim);
//        for(U s = 0; s < dim; s++)
//        {
//            REQUIRE( x[s] == Approx(x_goal[s]) );
//        }
//        REQUIRE( wgt == Approx(wgt_goal) );
//    };
//
//};
//
//TEST_CASE( "Trivial Transform", "[transform]")
//{
//    using D = double;
//    using U = unsigned long long int;
//
//    U dim = 9;
//
//    D x[]      = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
//    D wgt = 1.;
//
//    D x_goal[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
//    D wgt_goal = 1.;
//
//    SECTION( "Trivial" )
//    {
//        integrators::transforms::Trivial<D> transform;
//        transform(x,wgt,dim);
//        for(U s = 0; s < dim; s++)
//        {
//            REQUIRE( x[s] == Approx(x_goal[s]) );
//        }
//        REQUIRE( wgt == Approx(wgt_goal) );
//    };
//
//    SECTION( "Trivial" )
//    {
//        integrators::transforms::Trivial<D,U> transform;
//        transform(x,wgt,dim);
//        for(U s = 0; s < dim; s++)
//        {
//            REQUIRE( x[s] == Approx(x_goal[s]) );
//        }
//        REQUIRE( wgt == Approx(wgt_goal) );
//    };
//
//};
