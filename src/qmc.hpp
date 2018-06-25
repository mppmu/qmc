#ifndef QMC_H
#define QMC_H

#include <mutex>
#include <random> // mt19937_64, uniform_real_distribution
#include <vector>
#include <map>
#include <set>

// Custom Types
#include "types/logger.hpp"
#include "types/result.hpp"
#include "types/errormode.hpp"

namespace integrators
{
    template <typename T, typename D, typename U = unsigned long long int, typename G = std::mt19937_64>
    class Qmc
    {

    private:

        std::uniform_real_distribution<D> uniform_distribution = std::uniform_real_distribution<D>(0,1);

        void init_z(std::vector<U>& z, const U n, const U dim) const;
        void init_d(std::vector<D>& d, const U m, const U dim);
        void init_r(std::vector<T>& r, const U m, const U r_size_over_m) const;

        template <typename F1, typename F2> void worker(const U thread_id,U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m,  F1& func, const U dim, F2& integral_transform, const int device, D& time_in_ns, U& points_computed) const;
        template <typename F1, typename F2> result<T,U> sample(F1& func, const U dim, F2& integral_transform, const U n, const U m, std::vector<result<T,U>> & previous_iterations);
        void update(result<T,U>& res, U& n, U& m, U& function_evaluations) const;

    public:

        Logger logger;
        G randomgenerator;
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

        U get_next_n(U preferred_n) const;

        template <typename F1, typename F2> result<T,U> integrate(F1& func, const U dim, F2& integral_transform);
        template <typename F1> result<T,U> integrate(F1& func, const U dim);

        Qmc();
        virtual ~Qmc() {}
    };
};

// Implementation
#include "math/mul_mod.hpp"
#include "overloads/real.hpp"
#include "overloads/complex.hpp"
#include "transforms/korobov.hpp"
#include "transforms/tent.hpp"
#include "transforms/trivial.hpp"
#include "generatingvectors/cbcpt_dn1_100.hpp"
#include "generatingvectors/cbcpt_dn2_6.hpp"
#include "generatingvectors/cbcpt_cfftw1_6.hpp"
#include "core/cuda/compute_kernel.hpp"
#include "core/cuda/compute.hpp"
#include "core/cuda/setup.hpp"
#include "core/cuda/teardown.hpp"
#include "core/cuda/get_device_count.hpp"
#include "core/generic/compute.hpp"
#include "core/reduce.hpp"
#include "members.hpp"

#endif
