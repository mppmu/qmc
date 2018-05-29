#pragma once
#ifndef QMC_H
#define QMC_H

#include <mutex>
#include <vector>
#include <map>
#include <memory> // unique_ptr
#include <set>
#include <random> // mt19937_64, uniform_real_distribution
#include <type_traits> // make_signed
#include <iterator>
#include <functional> // reference_wrapper

namespace integrators
{

    enum ErrorMode
    {
        all,
        largest
    };

    struct Logger : public std::reference_wrapper<std::ostream>
    {
        using std::reference_wrapper<std::ostream>::reference_wrapper;
        template<typename T> std::ostream& operator<<(T arg) const { return this->get() << arg; }
        std::ostream& operator<<(std::ostream& (*arg)(std::ostream&)) const { return this->get() << arg; }
    };

    template <typename T, typename U = unsigned long long int>
    struct result
    {
        T integral;
        T error;
        U n;
        U m;
    };

// TODO - unsigned int MAXDIM = 20,
    template <typename T, typename D, typename U = unsigned long long int, typename G = std::mt19937_64>
    class Qmc
    {

    private:

        std::uniform_real_distribution<D> uniformDistribution = std::uniform_real_distribution<D>(0,1);

        void init_g();
        void init_z(std::vector<U>& z, const U n, const U dim) const;
        void init_d(std::vector<D>& d, const U m, const U dim);
        void init_r(std::vector<T>& r, const U m, const U r_size) const;
        
        result<T,U> reduce(const std::vector<T>& r, const U n, const U m, std::vector<result<T,U>> & previous_iterations);
        template <typename F1, typename F2> void compute(const int i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size, const U total_work_packages, const U n, const U m, F1& func, const U dim, F2& integral_transform);
        template <typename F1, typename F2> void compute_worker(const U thread_id, U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m,  F1& func, const U dim, F2& integral_transform, const int device);
#ifdef __CUDACC__
        template <typename F1, typename F2> void compute_gpu(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size, const U work_this_iteration, const U total_work_packages, const U n, const U m, F1* d_func, const U dim, F2* d_integral_transform, const int device, const U cudablocks, const U cudathreadsperblock);
#endif
        template <typename F1, typename F2> result<T,U> sample(F1& func, const U dim, F2& integral_transform, const U n, const U m, std::vector<result<T,U>> & previous_iterations);
        void update(result<T,U>& res, U& n, U& m, std::vector<result<T,U>> & previous_iterations);

    public:

        Logger logger;

        G randomgenerator;

        U minn;
        U minm;
        D epsrel;
        D epsabs;
        D border;
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

        template <typename F1, typename F2> result<T,U> integrate(F1& func, const U dim, F2& integral_transform);
        template <typename F1> result<T,U> integrate(F1& func, const U dim);
        U get_next_n(U preferred_n) const;

        Qmc();
        ~Qmc() {}

    };
    
};

// Implementation
#include "qmc_mul_mod.hpp"
#include "qmc_real.hpp"
#include "qmc_complex.hpp"
#include "qmc_transform.hpp"
#include "qmc_generating_vectors.hpp"
#ifdef __CUDACC__
#include "qmc_core_gpu.hpp"
#endif
#include "qmc_core.hpp"

#endif
