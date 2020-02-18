/*
 * Compile without GPU support:
 *   c++ -std=c++11 -O3 -pthread -I../src 1000_genz_demo.cpp -o 1000_genz_demo.out -lgsl -lgslcblas -lcuba -lm
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -O3 -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 1000_genz_demo.cpp -o 1000_genz_demo.out -lgsl -lgslcblas -lcuba -lm
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <cmath> // sin, cos, exp, acos, nan, erf
#include <algorithm>
#include <iterator>
#include <numeric>
#include <chrono>
#include <vector>
#include <bitset>

#include "qmc.hpp"
#include "cuba.h"

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#define NCOMP 1

// Integrator options
struct options_t {
    using U = unsigned long long int;
    using D = double;

    // Generic options
    D epsrel = 1e-8;
    D divonneepsrel = 1e-5;
    D epsabs = 0;

    // QMC Options
    U minn = 1;
    U qmcmaxeval     = 700000000; // 700000000 // (max lattice size)
    U vegasmaxeval   = 700000000;
    U divonnemaxeval = 700000000;
    U suavemaxeval   = 80000000;
    U cuhremaxeval   = 700000000;
    U verbosity = 0;

    // Cuba options
    // int ncomp = 1; // hardcoded to 1
    int flags = 0;
    int flagslast = 0 + 4; // 4 - use result of last iteration only
    int nvec = 1;
    int seed = 0;
    long long int mineval = 0;
    // long long int maxeval; // set above
    long long int nstart = 1000;
    long long int nincrease = 500;
    long long int nbatch = 1000;
    int gridno = 0;
    char* statefile = nullptr;
    int* spin = nullptr;
    long long int nnew = 1000;
    long long int nmin = 2;
    double flatness = 50;
    int key = 0;
    int key1 = -200; // Cuba Test Suite (T. Hahn), testsuite.m
    int key2 = 1;
    int key3 = 1;
    int maxpass = 5;
    double border = 0.;
    double maxchisq = 10.;
    double mindeviation = .25;
    int ngiven = 0;
    // int ldxgiven = ndim;
    long long int nextra = 0;
};

std::mt19937_64 randomgenerator( std::random_device{}() );
std::uniform_real_distribution<double> uniform_distribution{0,1};

// Integrand for QMC
template<integrators::U NDIM, int FAM>
struct family_t  {
    const unsigned long long int number_of_integration_variables = NDIM;
    double c[NDIM] = {0.};
    double w[NDIM] = {0.};
    const double pi = acos( -1.);
    // Call operator for qmc
    HOSTDEVICE double operator()(const double* x) const {
        if(FAM == 0)
        {
            double arg = 2.*pi*w[0];
            for(size_t i = 0; i<number_of_integration_variables; i++) arg += c[i]*x[i];
            return cos(arg);
        } else if (FAM == 1)
        {
            double res = 1.;
            for(size_t i = 0; i<number_of_integration_variables; i++) res *= 1./( (x[i]-w[i])*(x[i]-w[i]) + 1./(c[i]*c[i]) );
            return res;
        } else if (FAM == 2)
        {
            double arg = 1.;
            for(size_t i = 0; i<number_of_integration_variables; i++) arg += c[i]*x[i];
            return pow(1./arg,number_of_integration_variables+1);
        } else if (FAM == 3)
        {
            double arg = 0.;
            for(size_t i = 0; i<number_of_integration_variables; i++) arg -= c[i]*c[i]*(x[i]-w[i])*(x[i]-w[i]);
            return exp(arg);
        } else if (FAM == 4)
        {
            double arg = 0.;
            for(size_t i = 0; i<number_of_integration_variables; i++) arg -= c[i]*abs(x[i]-w[i]);
            return exp(arg);
        } else if (FAM == 5)
        {
            if(x[0] > w[0])
                return 0.;
            if(x[1] > w[1])
                return 0.;
            double arg = 0.;
            for(size_t i = 0; i<number_of_integration_variables; i++) arg += c[i]*x[i];
            return exp(arg);
        } else
        {
            return std::nan("1");
        }
    }
};

// Integrand for Cuba (wraps QMC integrand)
template<integrators::U NDIM, int FAM>
int cuba_integrand(const int *ndim, const double xx[],const int *ncomp, double ff[], void *userdata)
{
    family_t<NDIM,FAM> integrand = * static_cast<family_t<NDIM,FAM> *>(userdata);
    ff[0] = integrand(xx);
    return 0;
}

// Old method for computing family 2
//std::vector<std::vector<int>> combinations(const std::vector<int>& elements, const size_t r)
//{
//    std::vector<std::vector<int>> res;
//    // res.reserve(r); // note: must reserve (n r) to remove inefficiency
//    size_t n = elements.size();
//    std::vector<bool> v(n);
//    std::fill(v.begin(), v.begin()+r, true);
//    do {
//        std::vector<int> part;
//        part.reserve(r);
//        for (size_t i = 0; i < n; ++i) {
//            if (v[i]) part.push_back(elements.at(i));
//        }
//        res.push_back(part);
//    } while (std::prev_permutation(v.begin(), v.end()));
//    return res;
//}

template<integrators::U NDIM, int FAM>
double integrate_analytic(family_t<NDIM,FAM>& integrand)
{
    if(FAM == 0)
    {
        const double pi = acos( -1.);
        double arg;
        double res;
        arg = 2.*pi*integrand.w[0];
        for(size_t i = 0; i<integrand.number_of_integration_variables; i++) arg += 1./2.*integrand.c[i];
        res = cos(arg);
        for(size_t i = 0; i<integrand.number_of_integration_variables; i++) res *= 2.*sin(integrand.c[i]/2.)/integrand.c[i];
        return res;

    } else if (FAM == 1)
    {
        double res = 1.;
        for(size_t i = 0; i<integrand.number_of_integration_variables; i++) res *= integrand.c[i]*(atan(integrand.c[i]-integrand.c[i]*integrand.w[i]) + atan(integrand.c[i]*integrand.w[i]));
        return res;
    } else if (FAM == 2)
    {
        // Old method for computing family 2
//        double denominator = 1.;
//        for(size_t i = 0; i<integrand.number_of_integration_variables; i++) denominator *= (i+1.)*integrand.c[i];
//        std::vector<int> range_dim(NDIM);
//        std::iota(range_dim.begin(), range_dim.end(), 0);
//        double numerator = 0.;
//        for(size_t i = 1; i<integrand.number_of_integration_variables+1; i++)
//        {
//            std::vector<std::vector<int>> combs = combinations(range_dim,i);
//            for(size_t j = 0; j < combs.size(); j++)
//            {
//                std::vector<int> current_comb = combs.at(j);
//                double arg = 1.;
//                for(size_t k = 0; k < current_comb.size(); k++)
//                {
//                    arg += integrand.c[current_comb[k]];
//                }
//                numerator += pow(-1.,i)/arg;
//            }
//        }
//        return (1.+numerator)/denominator;
        double denominator = 1.;
        for(size_t i = 0; i<integrand.number_of_integration_variables; i++) denominator *= (i+1.)*integrand.c[i];
        double numerator = 0.;
        for(size_t i = 0; i<pow(2,integrand.number_of_integration_variables); i++)
        {
            double arg = 1.;
            std::bitset<NDIM> r(i);
            for(size_t j = 0; j<integrand.number_of_integration_variables; j++)
            {
                arg += integrand.c[j]*static_cast<int>(r[j]);
            }
            numerator += pow(-1.,r.count())/arg;
        }
        return numerator/denominator;
    } else if (FAM == 3)
    {
        const double pi = acos( -1.);
        double res = 1.;
        for(size_t i = 0; i<integrand.number_of_integration_variables; i++)
            res *= (sqrt(pi)/integrand.c[i]/2.)*( erf((1.-integrand.w[i]) * integrand.c[i]) -erf(-integrand.w[i] * integrand.c[i]));
        return res;
    } else if (FAM == 4)
    {
        double res = 1.;
        for(size_t i = 0; i<integrand.number_of_integration_variables; i++) res *= (2.-exp(-integrand.c[i]*integrand.w[i])-exp(integrand.c[i]*integrand.w[i]-integrand.c[i]))/integrand.c[i];
        return res;
    } else if (FAM == 5)
    {
        double res = 1.;
        for(size_t i = 0; i<2; i++) res *= ( -1. + exp(integrand.c[i]*integrand.w[i]))/integrand.c[i];
        for(size_t i = 2; i<integrand.number_of_integration_variables; i++) res *= (exp(integrand.c[i])-1.)/integrand.c[i];
        return res;
    } else
    {
        return std::nan("1");
    }
}

template<integrators::U NDIM, int FAM>
family_t<NDIM,FAM> generate_integrand()
{
    std::vector<double> difficulty = {6.0, 18.0, 2.2, 15.2, 16.1, 16.4}; // Cuba Test Suite (T. Hahn)
    double w[NDIM];
    double c[NDIM];
    for (size_t i = 0; i < NDIM; i++) w[i] = uniform_distribution(randomgenerator);
    for (size_t i = 0; i < NDIM; i++) c[i] = uniform_distribution(randomgenerator);
    double c_sum = std::accumulate(c, c+NDIM, 0.);
    for (size_t i = 0; i < NDIM; i++) c[i] *= difficulty[FAM]/c_sum;
    family_t<NDIM,FAM> integrand;
    std::copy(std::begin(c), std::end(c), std::begin(integrand.c));
    std::copy(std::begin(w), std::end(w), std::begin(integrand.w));
    return integrand;
}

void print_test_results(std::vector<double>& mean_correct_digits, std::vector<double>& mean_evaluations, std::vector<double>& mean_time)
{
    const char separator       = ' ';
    const int name_width       = 25;
    const int num_width        = 25;

    std::cout << std::left << std::setw(name_width) << std::setfill(separator) << "# Mean Evaluations";
    std::cout << std::left << std::setw(name_width) << std::setfill(separator) << "Mean Correct Digits";
    std::cout << std::left << std::setw(name_width) << std::setfill(separator) << "Mean Time (ms)";
    std::cout << std::endl;

    size_t num_integrators = mean_correct_digits.size();
    for(size_t i = 0; i < num_integrators; i++)
    {
        std::cout << std::left << std::setw(num_width) << std::setfill(separator) << mean_evaluations.at(i);
        std::cout << std::left << std::setw(num_width) << std::setfill(separator) << mean_correct_digits.at(i);
        std::cout << std::left << std::setw(num_width) << std::setfill(separator) << mean_time.at(i);
        std::cout << std::endl;
    }
}

template<integrators::U NDIM, int FAM>
void test(options_t& integrator_options)
{
    size_t iterations = 10;

    integrators::result<double> res;
    std::vector<double> mean_correct_digits = {0.,0.,0.,0.,0.,0.};
    std::vector<double> mean_evaluations = {0.,0.,0.,0.,0.,0.};
    std::vector<double> mean_time = {0.,0.,0.,0.,0.,0.};

    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    std::chrono::milliseconds diff_ms;

    // Setup CPU QMC
    const unsigned long long int MAXVAR = 10;
    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::None::type> qmc_cpu_integrator;
    qmc_cpu_integrator.epsrel = integrator_options.epsrel;
    qmc_cpu_integrator.epsabs = integrator_options.epsabs;
    qmc_cpu_integrator.minn = integrator_options.minn;
    qmc_cpu_integrator.devices = {-1};
    qmc_cpu_integrator.maxeval = integrator_options.qmcmaxeval;
    qmc_cpu_integrator.verbosity = integrator_options.verbosity;
    qmc_cpu_integrator.randomgenerator.seed(integrator_options.seed);

    // Setup CPU/GPU QMC
    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::None::type> qmc_integrator;
    qmc_integrator.epsrel = integrator_options.epsrel;
    qmc_integrator.epsabs = integrator_options.epsabs;
    qmc_integrator.minn = integrator_options.minn;
    qmc_integrator.maxeval = integrator_options.qmcmaxeval;
    qmc_integrator.verbosity = integrator_options.verbosity;
    qmc_integrator.randomgenerator.seed(integrator_options.seed);

    // Variables for Cuba
    long long int neval;
    int nregions, fail;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];

    // Variables for analytic calculation
    double analytic_result;
    double correct_digits;

    for(size_t i = 0; i < iterations; i++)
    {

        family_t<NDIM,FAM> integrand = generate_integrand<NDIM,FAM>();

        std::cout << "-- Integrand (NDIM " << NDIM << ", Family " << FAM << ", Iteration " << i << ") -- " << std::endl;
        std::cout << "c: ";
        for(size_t j = 0; j < NDIM; j++)
            std::cout << integrand.c[j] << " ";
        std::cout << std::endl;
        std::cout << "w: ";
        for(size_t j = 0; j < NDIM; j++)
            std::cout << integrand.w[j] << " ";
        std::cout << std::endl;

        // Analytic Result
        analytic_result = integrate_analytic<NDIM,FAM>(integrand);
        std::cout << "ANA RESULT:" << analytic_result << std::endl;

        // QMC CPU
        start = std::chrono::steady_clock::now();
        res = qmc_cpu_integrator.integrate(integrand);
        end = std::chrono::steady_clock::now();
        diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        correct_digits = -log10( abs((res.integral-analytic_result)/analytic_result));
        std::cout << "CPU RESULT:" << res.integral << " " << res.error << " " << abs(res.integral - analytic_result) << " " << correct_digits << " " << res.evaluations << " " << " " << diff_ms.count() << " # n*m = " << res.n*res.m  << " iterations = " << res.iterations << std::endl;
        mean_correct_digits.at(0) += correct_digits/static_cast<double>(iterations);
        mean_evaluations.at(0) += static_cast<double>(res.n)*static_cast<double>(res.m)/static_cast<double>(iterations);
        mean_time.at(0) += static_cast<double>(diff_ms.count())/static_cast<double>(iterations);

        // QMC CPU/GPU
        start = std::chrono::steady_clock::now();
        res = qmc_integrator.integrate(integrand);
        end = std::chrono::steady_clock::now();
        diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        correct_digits = -log10( abs((res.integral-analytic_result)/analytic_result));
        std::cout << "GPU RESULT:" << res.integral << " " << res.error << " " << abs(res.integral - analytic_result) << " " << correct_digits << " " << res.evaluations << " " << " " << diff_ms.count() << " # n*m = " << res.n*res.m  << " iterations = " << res.iterations << std::endl;
        mean_correct_digits.at(1) += correct_digits/static_cast<double>(iterations);
        mean_evaluations.at(1) += static_cast<double>(res.n)*static_cast<double>(res.m)/static_cast<double>(iterations);
        mean_time.at(1) += static_cast<double>(diff_ms.count())/static_cast<double>(iterations);

        // Vegas
        start = std::chrono::steady_clock::now();
        llVegas(NDIM,
                NCOMP,
                *cuba_integrand<NDIM,FAM>, &integrand,
                integrator_options.nvec,
                integrator_options.epsrel,
                integrator_options.epsabs,
                integrator_options.flags,
                integrator_options.seed,
                integrator_options.mineval,
                integrator_options.vegasmaxeval,
                integrator_options.nstart,
                integrator_options.nincrease,
                integrator_options.nbatch,
                integrator_options.gridno,
                integrator_options.statefile,
                integrator_options.spin,
                &neval, &fail, integral, error, prob);
        end = std::chrono::steady_clock::now();
        diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        correct_digits = -log10( abs(((double)integral[0]-analytic_result)/analytic_result));
        std::cout << "VEG RESULT:" << (double)integral[0] << " " << (double)error[0] << " " << abs((double)integral[0]-analytic_result) << " " << correct_digits << " " << neval << " " << diff_ms.count() << " # prob = " << (double)prob[0] << " fail =" << fail << std::endl;
        mean_correct_digits.at(2) += correct_digits/static_cast<double>(iterations);
        mean_evaluations.at(2) += static_cast<double>(neval)/static_cast<double>(iterations);
        mean_time.at(2) += static_cast<double>(diff_ms.count())/static_cast<double>(iterations);

        // Suave
        start = std::chrono::steady_clock::now();
        llSuave(NDIM,
                NCOMP,
                *cuba_integrand<NDIM,FAM>, &integrand,
                integrator_options.nvec,
                integrator_options.epsrel,
                integrator_options.epsabs,
                integrator_options.flagslast,
                integrator_options.seed,
                integrator_options.mineval,
                integrator_options.suavemaxeval,
                integrator_options.nnew,
                integrator_options.nmin,
                integrator_options.flatness,
                integrator_options.statefile,
                integrator_options.spin,
                &nregions, &neval, &fail, integral, error, prob);
        end = std::chrono::steady_clock::now();
        diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        correct_digits = -log10( abs(((double)integral[0]-analytic_result)/analytic_result));
        std::cout << "SUA RESULT:" << (double)integral[0] << " " << (double)error[0] << " " << abs((double)integral[0]-analytic_result) << " " << correct_digits << " " << neval << " " << diff_ms.count() << " # prob = " << (double)prob[0] << " fail =" << fail << std::endl;
        mean_correct_digits.at(3) += correct_digits/static_cast<double>(iterations);
        mean_evaluations.at(3) += static_cast<double>(neval)/static_cast<double>(iterations);
        mean_time.at(3) += static_cast<double>(diff_ms.count())/static_cast<double>(iterations);

        // Divonne
        start = std::chrono::steady_clock::now();
        llDivonne(NDIM,
                  NCOMP,
                  *cuba_integrand<NDIM,FAM>, &integrand,
                  integrator_options.nvec,
                  integrator_options.divonneepsrel,
                  integrator_options.epsabs,
                  integrator_options.flags,
                  integrator_options.seed,
                  integrator_options.mineval,
                  integrator_options.divonnemaxeval,
                  integrator_options.key1,
                  integrator_options.key2,
                  integrator_options.key3,
                  integrator_options.maxpass,
                  integrator_options.border,
                  integrator_options.maxchisq,
                  integrator_options.mindeviation,
                  integrator_options.ngiven,
                  NDIM, // ldxgiven
                  nullptr, // xgiven
                  integrator_options.nextra,
                  nullptr, // peakfinder
                  integrator_options.statefile,
                  integrator_options.spin,
                  &nregions, &neval, &fail, integral, error, prob);
        end = std::chrono::steady_clock::now();
        diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        correct_digits = -log10( abs(((double)integral[0]-analytic_result)/analytic_result));
        std::cout << "DIV RESULT:" << (double)integral[0] << " " << (double)error[0] << " " << abs((double)integral[0]-analytic_result) << " " << correct_digits << " " << neval << " " << diff_ms.count() << " # prob = " << (double)prob[0] << " fail =" << fail << std::endl;
        mean_correct_digits.at(4) += correct_digits/static_cast<double>(iterations);
        mean_evaluations.at(4) += static_cast<double>(neval)/static_cast<double>(iterations);
        mean_time.at(4) += static_cast<double>(diff_ms.count())/static_cast<double>(iterations);

        // Cuhre
        start = std::chrono::steady_clock::now();
        llCuhre(NDIM,
                NCOMP,
                *cuba_integrand<NDIM,FAM>, &integrand,
                integrator_options.nvec,
                integrator_options.epsrel,
                integrator_options.epsabs,
                integrator_options.flagslast,
                integrator_options.mineval,
                integrator_options.cuhremaxeval,
                integrator_options.key,
                integrator_options.statefile,
                integrator_options.spin,
                &nregions, &neval, &fail, integral, error, prob);
        end = std::chrono::steady_clock::now();
        diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        correct_digits = -log10( abs(((double)integral[0]-analytic_result)/analytic_result));
        std::cout << "CUH RESULT:" << (double)integral[0] << " " << (double)error[0] << " " << abs((double)integral[0]-analytic_result) << " " << correct_digits << " " << neval << " " << diff_ms.count() << " # prob = " << (double)prob[0] << " fail =" << fail << std::endl;
        mean_correct_digits.at(5) += correct_digits/static_cast<double>(iterations);
        mean_evaluations.at(5) += static_cast<double>(neval)/static_cast<double>(iterations);
        mean_time.at(5) += static_cast<double>(diff_ms.count())/static_cast<double>(iterations);
    }

    std::cout << "-- Final Results (NDIM " << NDIM << ", Family " << FAM << ") -- " << std::endl;
    print_test_results(mean_correct_digits, mean_evaluations, mean_time);

}

void do_test(options_t& options)
{

    std::cout << "# BEGIN WARMUP " << std::endl;
    options_t options_warmup;
    options_warmup.qmcmaxeval = 2000;
    options_warmup.vegasmaxeval = 2000;
    options_warmup.divonnemaxeval = 2000;
    options_warmup.suavemaxeval = 2000;
    options_warmup.cuhremaxeval = 2000;
    test<5,0>(options_warmup);
    std::cout << "# WARMUP FINISHED" << std::endl;

    test<5,0>(options);
    test<8,0>(options);
    test<10,0>(options);

    test<5,1>(options);
    test<8,1>(options);
    test<10,1>(options);

    test<5,2>(options);
    test<8,2>(options);
    test<10,2>(options);

    test<5,3>(options);
    test<8,3>(options);
    test<10,3>(options);

    test<5,4>(options);
    test<8,4>(options);
    test<10,4>(options);

    test<5,5>(options);
    test<8,5>(options);
    test<10,5>(options);
}

int main()
{

    // Set up random generator
    randomgenerator.seed(0);

    // Set up integrator options
    options_t options;

    // Set ouput precision
    std::cout << std::setprecision(18) << std::endl;

    // Run test suite
    do_test(options);
}
