#include "catch.hpp"
#include "qmc.hpp"

TEST_CASE( "cbcpt_cfftw1_6", "[generatingvectors]")
{
    using U = unsigned long long int;

    std::map<U,std::vector<U>> gvec = integrators::generatingvectors::cbcpt_cfftw1_6();

    SECTION("Length")
    {
        REQUIRE( gvec.lower_bound(1)->second.size() == 6);
    };

};

TEST_CASE( "cbcpt_dn1_100", "[generatingvectors]")
{
    using U = unsigned long long int;

    std::map<U,std::vector<U>> gvec = integrators::generatingvectors::cbcpt_dn1_100();

    SECTION("Length")
    {
        REQUIRE( gvec.lower_bound(1)->second.size() == 100);
    };

};

TEST_CASE( "cbcpt_dn2_6", "[generatingvectors]")
{
    using U = unsigned long long int;

    std::map<U,std::vector<U>> gvec = integrators::generatingvectors::cbcpt_dn2_6();

    SECTION("Length")
    {
        REQUIRE( gvec.lower_bound(1)->second.size() == 6);
    };

};
