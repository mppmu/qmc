#include "catch.hpp"
#include "qmc.hpp"

#include <complex>
#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

TEST_CASE( "Real", "[overloads]")
{
    using T = double;

    T mean = 15.;
    T variance = 10.;
    T sum = 1.;
    T delta = 0.1;

    T compute_variance_target = 8.6;
    T compute_error_target = 3.1622776601683793320;
    T compute_variance_from_error_target = 10.;

    integrators::result<T> res = {10.,1.,7,2,1,14};

    SECTION("compute_variance")
    {
        REQUIRE( integrators::overloads::compute_variance(mean,variance,sum,delta) == Approx(compute_variance_target) );
    };

    SECTION("compute_error")
    {
        REQUIRE( integrators::overloads::compute_error(variance) == Approx(compute_error_target) );
    };

    SECTION("compute_variance_from_error (positive)")
    {
        REQUIRE( integrators::overloads::compute_variance_from_error(compute_error_target) == Approx(compute_variance_from_error_target) );
    };

    SECTION("compute_variance_from_error (negative)")
    {
        REQUIRE( integrators::overloads::compute_variance_from_error(-compute_error_target) == Approx(compute_variance_from_error_target) );
    };

    SECTION("compute_error_ratio (epsrel, all)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::all;

        double epsrel = 1e-3;
        double epsabs = 1e-5;
        double compute_error_ratio_target = 100.;
        REQUIRE( integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target) );
    };

    SECTION("compute_error_ratio (epsabs, all)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::all;

        double epsrel = 1e-5;
        double epsabs = 1e-3;
        double compute_error_ratio_target = 1000.;
        REQUIRE( integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target) );
    };

    SECTION("compute_error_ratio (epsrel, largest)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::largest;

        double epsrel = 1e-3;
        double epsabs = 1e-5;
        double compute_error_ratio_target = 100.;
        REQUIRE( integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target) );
    };

    SECTION("compute_error_ratio (epsabs, largest)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::largest;

        double epsrel = 1e-5;
        double epsabs = 1e-3;
        double compute_error_ratio_target = 1000.;
        REQUIRE( integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target) );
    };

};

TEST_CASE( "std::complex", "[overloads]")
{
    using T = std::complex<double>;

    T mean = {15.,20.};
    T variance = {10.,30.};
    T sum = {1.,2.};
    T delta = {0.1,0.2};

    T compute_variance_target = {8.6,26.4};
    T compute_error_target = {3.1622776601683793320,5.4772255750516611346};
    T compute_variance_from_error_target = {10.,30.};

    integrators::result<T> res_large_error_real = {{10.,2.},{1.,0.05},7,2,1,14};
    integrators::result<T> res_large_error_imag = {{10.,100.},{1.,15.},7,2,1,14};
    integrators::result<T> res_mixed1 = {{10.,100.},{15.,1.},7,2,1,14};
    integrators::result<T> res_mixed2 = {{100.,10.},{1.,15.},7,2,1,14};

    SECTION("compute_variance")
    {
        REQUIRE( integrators::overloads::compute_variance(mean,variance,sum,delta).real() == Approx(compute_variance_target.real()) );
        REQUIRE( integrators::overloads::compute_variance(mean,variance,sum,delta).imag() == Approx(compute_variance_target.imag()) );
    };

    SECTION("compute_error")
    {
        REQUIRE( integrators::overloads::compute_error(variance).real() == Approx(compute_error_target.real()) );
        REQUIRE( integrators::overloads::compute_error(variance).imag() == Approx(compute_error_target.imag()) );
    };

    SECTION("compute_variance_from_error (positive)")
    {
        REQUIRE( integrators::overloads::compute_variance_from_error(compute_error_target).real() == Approx(compute_variance_from_error_target.real()) );
        REQUIRE( integrators::overloads::compute_variance_from_error(compute_error_target).imag() == Approx(compute_variance_from_error_target.imag()) );
    };

    SECTION("compute_variance_from_error (negative)")
    {
        REQUIRE( integrators::overloads::compute_variance_from_error(-compute_error_target).real() == Approx(compute_variance_from_error_target.real()) );
        REQUIRE( integrators::overloads::compute_variance_from_error(-compute_error_target).imag() == Approx(compute_variance_from_error_target.imag()) );
    };

    SECTION("compute_error_ratio (epsrel, all)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::all;

        double epsrel = 1e-3;
        double epsabs = 1e-5;
        double compute_error_ratio_target_real = 100.;
        double compute_error_ratio_target_imag = 150.;
        double compute_error_ratio_mixed1 = 1500.;
        double compute_error_ratio_mixed2 = 1500.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio (epsabs, all)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::all;

        double epsrel = 1e-5;
        double epsabs = 1e-3;
        double compute_error_ratio_target_real = 1000.;
        double compute_error_ratio_target_imag = 15000.;
        double compute_error_ratio_mixed1 = 15000.;
        double compute_error_ratio_mixed2 = 15000.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio (epsrel, largest)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::largest;

        double epsrel = 1e-3;
        double epsabs = 1e-5;
        double compute_error_ratio_target_real = 100.;
        double compute_error_ratio_target_imag = 150.;
        double compute_error_ratio_mixed1 = 150.;
        double compute_error_ratio_mixed2 = 150.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio (epsrel, largest)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::largest;

        double epsrel = 1e-5;
        double epsabs = 1e-3;
        double compute_error_ratio_target_real = 1000.;
        double compute_error_ratio_target_imag = 15000.;
        double compute_error_ratio_mixed1 = 15000.;
        double compute_error_ratio_mixed2 = 15000.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio with invalid errormode")
    {
        integrators::ErrorMode errormode = static_cast<integrators::ErrorMode>(-1); // invalid enum

        double epsrel = 1e-5;
        double epsabs = 1e-3;

        REQUIRE_THROWS_AS( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode), std::invalid_argument);
    };

};

#ifdef __CUDACC__
TEST_CASE( "thrust::complex", "[overloads]")
{
    using T = thrust::complex<double>;

    T mean = {15.,20.};
    T variance = {10.,30.};
    T sum = {1.,2.};
    T delta = {0.1,0.2};

    T compute_variance_target = {8.6,26.4};
    T compute_error_target = {3.1622776601683793320,5.4772255750516611346};
    T compute_variance_from_error_target = {10.,30.};

    integrators::result<T> res_large_error_real = {{10.,2.},{1.,0.05},7,2,1,14};
    integrators::result<T> res_large_error_imag = {{10.,100.},{1.,15.},7,2,1,14};
    integrators::result<T> res_mixed1 = {{10.,100.},{15.,1.},7,2,1,14};
    integrators::result<T> res_mixed2 = {{100.,10.},{1.,15.},7,2,1,14};

    SECTION("compute_variance")
    {
        REQUIRE( integrators::overloads::compute_variance(mean,variance,sum,delta).real() == Approx(compute_variance_target.real()) );
        REQUIRE( integrators::overloads::compute_variance(mean,variance,sum,delta).imag() == Approx(compute_variance_target.imag()) );
    };

    SECTION("compute_error")
    {
        REQUIRE( integrators::overloads::compute_error(variance).real() == Approx(compute_error_target.real()) );
        REQUIRE( integrators::overloads::compute_error(variance).imag() == Approx(compute_error_target.imag()) );
    };

    SECTION("compute_variance_from_error (positive)")
    {
        REQUIRE( integrators::overloads::compute_variance_from_error(compute_error_target).real() == Approx(compute_variance_from_error_target.real()) );
        REQUIRE( integrators::overloads::compute_variance_from_error(compute_error_target).imag() == Approx(compute_variance_from_error_target.imag()) );
    };

    SECTION("compute_variance_from_error (negative)")
    {
        REQUIRE( integrators::overloads::compute_variance_from_error(-compute_error_target).real() == Approx(compute_variance_from_error_target.real()) );
        REQUIRE( integrators::overloads::compute_variance_from_error(-compute_error_target).imag() == Approx(compute_variance_from_error_target.imag()) );
    };

    SECTION("compute_error_ratio (epsrel, all)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::all;

        double epsrel = 1e-3;
        double epsabs = 1e-5;
        double compute_error_ratio_target_real = 100.;
        double compute_error_ratio_target_imag = 150.;
        double compute_error_ratio_mixed1 = 1500.;
        double compute_error_ratio_mixed2 = 1500.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio (epsabs, all)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::all;

        double epsrel = 1e-5;
        double epsabs = 1e-3;
        double compute_error_ratio_target_real = 1000.;
        double compute_error_ratio_target_imag = 15000.;
        double compute_error_ratio_mixed1 = 15000.;
        double compute_error_ratio_mixed2 = 15000.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio (epsrel, largest)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::largest;

        double epsrel = 1e-3;
        double epsabs = 1e-5;
        double compute_error_ratio_target_real = 100.;
        double compute_error_ratio_target_imag = 150.;
        double compute_error_ratio_mixed1 = 150.;
        double compute_error_ratio_mixed2 = 150.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio (epsrel, largest)")
    {
        integrators::ErrorMode errormode = integrators::ErrorMode::largest;

        double epsrel = 1e-5;
        double epsabs = 1e-3;
        double compute_error_ratio_target_real = 1000.;
        double compute_error_ratio_target_imag = 15000.;
        double compute_error_ratio_mixed1 = 15000.;
        double compute_error_ratio_mixed2 = 15000.;
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_real) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_large_error_imag, epsrel, epsabs, errormode) == Approx(compute_error_ratio_target_imag) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed1, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed1) );
        REQUIRE( integrators::overloads::compute_error_ratio(res_mixed2, epsrel, epsabs, errormode) == Approx(compute_error_ratio_mixed2) );
    };

    SECTION("compute_error_ratio with invalid errormode")
    {
        integrators::ErrorMode errormode = static_cast<integrators::ErrorMode>(-1); // invalid enum

        double epsrel = 1e-5;
        double epsabs = 1e-3;

        REQUIRE_THROWS_AS( integrators::overloads::compute_error_ratio(res_large_error_real, epsrel, epsabs, errormode), std::invalid_argument);
    };

};
#endif

