#include "catch.hpp"
#include "../../src/qmc_mul_mod.hpp"

#include <iostream>
#include <limits>
#include <type_traits>

TEST_CASE( "Modular Multiplication", "[mul_mod]" ) {

    // Print diagnostic information
    std::cout
    << "std::numeric_limits<unsigned long long int>::is_modulo" << " "
    << std::numeric_limits<unsigned long long int>::is_modulo
    << std::endl;

//    std::cout
//    << "std::numeric_limits<typename std::make_signed<unsigned long long int>::type>::max()" << " "
//    << std::numeric_limits<typename std::make_signed<unsigned long long int>::type>::max()
//    << std::endl;
//
//    auto old_cout_precision = std::cout.precision(std::numeric_limits<double>::max_digits10);
//    std::cout
//    << std::fixed
//    << "pow(std::numeric_limits<double>::radix,std::numeric_limits<double>::digits)" << " "
//    << pow(std::numeric_limits<double>::radix,std::numeric_limits<double>::digits)
//    << std::endl;
//    std::cout.precision(old_cout_precision);
//
//    unsigned long long int a = 4503599627370496ull; // pow(radix,digits-1)
//    unsigned long long int b = 4503599627370495ull;
//
//    unsigned long long int c = 3037050499ull;
//    unsigned long long int d = 4037000499ull;
//    unsigned long long int e = 6854775807ull;
//
//    unsigned long long int f = 27595940ull;
//    unsigned long long int g = 1401529ull;
//    unsigned long long int h = 3867651019227ull;
//
//    unsigned long long int zero = 0ull;
//
//    SECTION( "Maximum unsigned long long int")
//    {
//        unsigned long long int result;
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(b,b,a);
//        REQUIRE( result == 1ull );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(b,1ull,a);
//        REQUIRE( result == 4503599627370495ull );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(1ll,b,a);
//        REQUIRE( result == 4503599627370495ull );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(b,2ull,a);
//        REQUIRE( result == 4503599627370494ull );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(2ll,b,a);
//        REQUIRE( result == 4503599627370494ull );
//
//    };
//
//    SECTION( "a*b overflows long long int")
//    {
//        unsigned long long int result;
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(c,d,e);
//        REQUIRE( result == 93292437ull );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(d,c,e);
//        REQUIRE( result == 93292437ull );
//
//    };
//
//    SECTION( "Random unsigned long long int")
//    {
//        unsigned long long int result;
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(f,g,h);
//        REQUIRE( result == 3867651019217ull );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(g,f,h);
//        REQUIRE( result == 3867651019217ull );
//    };
//
//    SECTION( "Zero")
//    {
//        unsigned long long int result;
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(d,zero,e);
//        REQUIRE( result == zero );
//
//        result = integrators::mul_mod<unsigned long long int,double,unsigned long long int>(zero,d,e);
//        REQUIRE( result == zero );
//    };

};
