#ifndef QMC_H
#define QMC_H

#include <mutex>
#include <random> // mt19937_64, uniform_real_distribution
#include <vector>
#include <map>
#include <set>

#include <gsl/gsl_multifit_nlinear.h>

// Custom Types
#include "types/logger.hpp"
#include "types/result.hpp"
#include "types/samples.hpp"
#include "types/errormode.hpp"

namespace integrators
{
    template <typename T, typename D, typename U = unsigned long long int, typename G = std::mt19937_64, typename H = std::uniform_real_distribution<D>>
    class Qmc
    {

    private:

        H uniform_distribution{0,1};

        void init_z(std::vector<U>& z, const U n, const U dim) const;
        void init_d(std::vector<D>& d, const U m, const U dim);
        void init_r(std::vector<T>& r, const U m, const U r_size_over_m) const;

        template <typename F1> void sample_worker(const U thread_id,U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m,  F1& func, const int device, D& time_in_ns, U& points_computed) const;
        template <typename F1> void evaluate_worker(const U thread_id,U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U n, F1& func, const int device, D& time_in_ns, U& points_computed) const;
        template <typename F1> result<T,U> sample(F1& func, const U n, const U m, std::vector<result<T,U>> & previous_iterations);
        void update(const result<T,U>& res, U& n, U& m, U& function_evaluations) const;
        template <typename F1> result<T,U> integrate_impl(F1& func);

    public:

        Logger logger;
        G randomgenerator;
        bool defaulttransform;
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

        int fitmaxiter;
        double fitxtol;
        double fitgtol;
        double fitftol;
        gsl_multifit_nlinear_parameters fitparametersgsl;

        U get_next_n(U preferred_n) const;

        template <typename F1> result<T,U> integrate(F1& func);
        template <typename F1> samples<T,D,U> evaluate(F1& func); // TODO: explicit test cases for this function
        template <typename F1, typename F2, typename F3, typename F4> F4& fit(F1& func, F2& fit_function, F3& fit_function_jacobian, F4& fit_function_transform); // TODO: explicit test cases for this function (minnevaluate = 0)

        Qmc();
        virtual ~Qmc() {}
    };
};

// Implementation
#include "math/mul_mod.hpp"
#include "math/argsort.hpp"
#include "overloads/real.hpp"
#include "overloads/complex.hpp"
#include "fitfunctions/polysingular.hpp"
#include "transforms/korobov.hpp"
#include "transforms/sidi.hpp"
#include "transforms/baker.hpp"
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
