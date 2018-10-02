#ifndef QMC_H
#define QMC_H

#include <mutex>
#include <random> // mt19937_64, uniform_real_distribution
#include <vector>
#include <map>
#include <set>

#include <gsl/gsl_multifit_nlinear.h>

namespace integrators
{
    using U = unsigned long long int;
}

// Custom Types
#include "types/logger.hpp"
#include "types/result.hpp"
#include "types/samples.hpp"
#include "types/errormode.hpp"

// Fit Functions
#include "fitfunctions/none.hpp"
#include "fitfunctions/polysingular.hpp"

// Periodizing Transforms
#include "transforms/none.hpp"
#include "transforms/baker.hpp"
#include "transforms/korobov.hpp"
#include "transforms/sidi.hpp"

namespace integrators
{
    template <
                 typename T, typename D, U M,
                 template<typename,typename,U> class P = transforms::Korobov<3>::template type,
                 template<typename,typename,U> class F = fitfunctions::None::template type,
                 typename G = std::mt19937_64, typename H = std::uniform_real_distribution<D>
             >
    class Qmc
    {

    private:

        H uniform_distribution{0,1};

        void init_z(std::vector<U>& z, const U n, const U number_of_integration_variables) const;
        void init_d(std::vector<D>& d, const U m, const U number_of_integration_variables);
        void init_r(std::vector<T>& r, const U m, const U r_size_over_m) const;

        template <typename I> void sample_worker(const U thread_id,U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m,  I& func, const int device, D& time_in_ns, U& points_computed) const;
        template <typename I> void evaluate_worker(const U thread_id,U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U n, I& func, const int device, D& time_in_ns, U& points_computed) const;
        template <typename I> result<T> sample(I& func, const U n, const U m, std::vector<result<T>> & previous_iterations);
        void update(const result<T>& res, U& n, U& m, U& function_evaluations) const;
        template <typename I> result<T> integrate_no_fit_no_transform(I& func);

    public:

        Logger logger;
        G randomgenerator;
        U minnevaluate;
        U minn;
        U minm;
        D epsrel;
        D epsabs;
        U maxeval;
        U maxnperpackage;
        U maxmperpackage;
        ErrorMode errormode;
        U cputhreads;
        U cudablocks;
        U cudathreadsperblock;
        std::set<int> devices;
        std::map<U,std::vector<U>> generatingvectors;
        U verbosity;

        U fitstepsize;
        int fitmaxiter;
        double fitxtol;
        double fitgtol;
        double fitftol;
        gsl_multifit_nlinear_parameters fitparametersgsl;

        U get_next_n(U preferred_n) const;

        template <typename I> result<T> integrate(I& func);
        template <typename I> samples<T,D> evaluate(I& func); // TODO: explicit test cases for this function
        template <typename I> typename F<I,D,M>::transform_t fit(I& func); // TODO: explicit test cases for this function
        Qmc();
        virtual ~Qmc() {}
    };
};

// Implementation
#include "math/mul_mod.hpp"
#include "math/argsort.hpp"
#include "overloads/real.hpp"
#include "overloads/complex.hpp"
#include "generatingvectors/cbcpt_dn1_100.hpp"
#include "generatingvectors/cbcpt_dn2_6.hpp"
#include "generatingvectors/cbcpt_cfftw1_6.hpp"
#include "core/cuda/compute_kernel.hpp"
#include "core/cuda/compute.hpp"
#include "core/cuda/setup.hpp"
#include "core/cuda/teardown.hpp"
#include "core/cuda/get_device_count.hpp"
#include "core/generic/compute.hpp"
#include "core/least_squares.hpp"
#include "core/reduce.hpp"
#include "members.hpp"

#endif
