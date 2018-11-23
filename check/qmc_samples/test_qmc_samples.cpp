#include "catch.hpp"
#include "qmc.hpp"

TEST_CASE( "Samples", "[samples]") {

    unsigned long long int dim = 2;
    std::vector<unsigned long long int> z = {1,2};
    std::vector<double> d = {0.5,0.2};
    std::vector<double> r = {1.,2.,3.,4.,5.};
    unsigned long long int n = 5;
    std::vector<std::vector<double>> expected_x =
    {
        {0.5,0.2},
        {0.7,0.6},
        {0.9,0.},
        {0.1,0.4},
        {0.3,0.8}
    };

    integrators::samples<double,double> test_samples;
    test_samples.z = z;
    test_samples.d = d;
    test_samples.r = r;
    test_samples.n = n;

    SECTION( "Access Fields", "[samples]" )
    {

        for(unsigned long long int i = 0; i < dim; i++)
        {
            REQUIRE( test_samples.z.at(i) == z.at(i) );
            REQUIRE( test_samples.d.at(i) == d.at(i) );
        }
        for(unsigned long long int i = 0; i < n; i++)
        {
            REQUIRE( test_samples.r.at(i) == r.at(i) );
        }
        REQUIRE( test_samples.n == n );
    };

    SECTION( "get_x", "[samples]" )
    {
        for(unsigned long long int i = 0; i < n; i ++)
        {
            for(unsigned long long int j = 0; j < dim; j++)
            {
                REQUIRE( test_samples.get_x(i,j) == Approx(expected_x.at(i).at(j)) );
            }
        }

    }

};
