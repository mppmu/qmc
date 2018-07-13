#ifndef QMC_MEMBERS_H
#define QMC_MEMBERS_H

#include <cstddef> // size_t
#include <cmath> // modf, abs, sqrt, pow
#include <stdexcept> // domain_error, invalid_argument
#include <thread> // thread
#include <algorithm> // min, max
#include <type_traits> // make_signed
#include <limits> // numeric_limits
#include <string> // to_string
#include <vector>
#include <iostream>
#include <iterator> // advance
#include <mutex>
#include <memory> // unique_ptr
#include <cassert> // assert
#include <chrono>

namespace integrators
{
    template <typename T, typename D, typename U, typename G, typename H>
    void Qmc<T,D,U,G,H>::init_z(std::vector<U>& z, const U n, const U dim) const
    {
        z = generatingvectors.at(n);
        if ( dim > z.size() ) throw std::domain_error("dim > generating vector dimension. Please supply a generating vector table with a larger number of dimensions.");
        z.resize(dim);
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    void Qmc<T,D,U,G,H>::init_d(std::vector<D>& d, const U m, const U dim)
    {
        d.clear();
        for (U k = 0; k < m; k++)
            for (U sDim = 0; sDim < dim; sDim++)
                d.push_back(uniform_distribution(randomgenerator));
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    void Qmc<T,D,U,G,H>::init_r(std::vector<T>& r, const U m, const U r_size_over_m) const
    {
        r.clear();
        r.resize(m * r_size_over_m, {0.});
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    template <typename F1, typename F2>
    void Qmc<T,D,U,G,H>::worker(const U thread_id, U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m, F1& func, const U dim, F2& integral_transform, const int device, D& time_in_ns, U& points_computed) const
    {
        std::chrono::steady_clock::time_point time_before_compute = std::chrono::steady_clock::now();

        points_computed = 0;

        // Setup worker
#ifdef __CUDACC__
        // define device pointers (must be accessible in local scope of the entire function)
        U d_r_size = m*cudablocks*cudathreadsperblock;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<F1>> d_func;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<F2>> d_integral_transform;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>> d_z;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>> d_d;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>> d_r;
#endif
        U i;
        U  work_this_iteration;
        if (device == -1) {
            work_this_iteration = 1;
        } else {
            work_this_iteration = cudablocks*cudathreadsperblock;
#ifdef __CUDACC__
            integrators::core::cuda::setup(d_z, z, d_d, d, d_r, d_r_size/m, &r[thread_id], r.size()/m, m, d_func, func, d_integral_transform, integral_transform, device, verbosity, logger);
#endif
        }

        bool work_remaining = true;
        while( work_remaining )
        {
            // Get work
            work_queue_mutex.lock();
            if (work_queue == 0)
            {
                work_remaining=false;
                i = 0;
            }
            else if (work_queue >= work_this_iteration)
            {
                work_queue-=work_this_iteration;
                i = work_queue;
            }
            else
            {
                work_this_iteration = work_queue;
                work_queue = 0;
                i = 0;
            }
            work_queue_mutex.unlock();
            
            if( !work_remaining )
                break;
            
            // Do work
            if (device == -1)
            {
                integrators::core::generic::compute(i, z, d, &r[thread_id], r.size()/m, total_work_packages, n, m, func, dim, integral_transform);
            }
            else
            {
#ifdef __CUDACC__
                integrators::core::cuda::compute(*this, i, work_this_iteration, total_work_packages, d_z, d_d, d_r, d_r_size/m, n, m, d_func, dim, d_integral_transform, device);
#endif
            }

            points_computed += work_this_iteration*m;

        }

        // Teardown worker
#ifdef __CUDACC__
        if (device != -1) {
            integrators::core::cuda::teardown(d_r, d_r_size/m, &r[thread_id], r.size()/m, m, device, verbosity, logger);
        }
#endif

        std::chrono::steady_clock::time_point time_after_compute = std::chrono::steady_clock::now();
        time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_after_compute - time_before_compute).count();

    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    template <typename F1, typename F2>
    result<T,U> Qmc<T,D,U,G,H>::sample(F1& func, const U dim, F2& integral_transform, const U n, const U m, std::vector<result<T,U>> & previous_iterations)
    {
        std::vector<U> z;
        std::vector<D> d;
        std::vector<T> r;

        result<T,U> res;

        U points_per_package = std::min(maxnperpackage, n); // points to compute per thread per work_package
        U total_work_packages = n/points_per_package; // Set total number of work packages to be computed
        if( n%points_per_package != 0) total_work_packages++;

        U extra_threads = devices.size() - devices.count(-1);
        
        // Memory required for result vector
        U r_size_over_m = extra_threads*cudablocks*cudathreadsperblock; // non-cpu workers
        if (devices.count(-1) != 0)
        {
            r_size_over_m += cputhreads; // cpu-workers
        }

        U iterations = (m+maxmperpackage-1)/maxmperpackage;
        U shifts_per_iteration = std::min(m,maxmperpackage);
        for(U iteration = 0; iteration < iterations; iteration++)
        {
            U shifts = shifts_per_iteration;
            if ( iteration == iterations-1)
            {
                // last iteration => compute remaining shifts
                shifts = m%maxmperpackage == 0 ? std::min(m,maxmperpackage) : m%maxmperpackage;
            }

            // Generate z, d, r
            init_z(z, n, dim);
            init_d(d, shifts, dim);
            init_r(r, shifts, r_size_over_m);

            if (verbosity > 0)
            {
                logger << "-- qmc::sample called --" << std::endl;
                logger << "dim " << dim << std::endl;
                logger << "minn " << minn << std::endl;
                logger << "minm " << minm << std::endl;
                logger << "epsrel " << epsrel << std::endl;
                logger << "epsabs " << epsabs << std::endl;
                logger << "maxeval " << maxeval << std::endl;
                logger << "cputhreads " << cputhreads << std::endl;
                logger << "maxnperpackage " << maxnperpackage << std::endl;
                logger << "maxmperpackage " << maxmperpackage << std::endl;
                logger << "cudablocks " << cudablocks << std::endl;
                logger << "cudathreadsperblock " << cudathreadsperblock << std::endl;
                logger << "devices ";
                for (const int& i : devices)
                    logger << i << " ";
                logger << std::endl;
                logger << "n " << n << std::endl;
                logger << "m " << m << std::endl;
                logger << "shifts " << shifts << std::endl;
                logger << "iterations " << iterations << std::endl;
                logger << "total_work_packages " << total_work_packages << std::endl;
                logger << "points_per_package " << points_per_package << std::endl;
                logger << "r " << shifts << "*" << r_size_over_m << std::endl;
            }

            std::chrono::steady_clock::time_point time_before_compute = std::chrono::steady_clock::now();

            if ( cputhreads == 1 && devices.size() == 1 && devices.count(-1) == 1)
            {
                // Compute serially on cpu
                if (verbosity > 2) logger << "computing serially" << std::endl;
                for( U i=0; i < total_work_packages; i++)
                {
                    integrators::core::generic::compute(i, z, d, &r[0], r.size()/shifts, total_work_packages, n, shifts, func, dim, integral_transform);
                }
            }
            else
            {
                // Create threadpool
                if (verbosity > 2)
                {
                    logger << "distributing work" << std::endl;
                    if ( devices.count(-1) != 0)
                        logger << "creating " << std::to_string(cputhreads) << " cputhreads," << std::to_string(extra_threads) << " non-cpu threads" << std::endl;
                    else
                        logger << "creating " << std::to_string(extra_threads) << " non-cpu threads" << std::endl;
                }

                // Setup work queue
                std::mutex work_queue_mutex;
                U work_queue = total_work_packages;

                // Launch worker threads
                U thread_id = 0;
                U thread_number = 0;
                std::vector<std::thread> thread_pool;
                thread_pool.reserve(cputhreads+extra_threads);
                std::vector<D> time_in_ns_per_thread(cputhreads+extra_threads,D(0));
                std::vector<U> points_computed_per_thread(cputhreads+extra_threads,U(0));
                for (int device : devices)
                {
                    if( device != -1)
                    {
#ifdef __CUDACC__
                        thread_pool.push_back( std::thread( &Qmc<T,D,U,G,H>::worker<F1,F2>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), total_work_packages, n, shifts, std::ref(func), dim, std::ref(integral_transform), device, std::ref(time_in_ns_per_thread[thread_number]), std::ref(points_computed_per_thread[thread_number])  ) ); // Launch non-cpu workers
                        thread_id += cudablocks*cudathreadsperblock;
                        thread_number += 1;
#else
                        throw std::invalid_argument("qmc::sample called with device != -1 (CPU) but CUDA not supported by compiler, device: " + std::to_string(device));
#endif
                    }
                }
                if( devices.count(-1) != 0)
                {
                    for ( U i=0; i < cputhreads; i++)
                    {
                        thread_pool.push_back( std::thread( &Qmc<T,D,U,G,H>::worker<F1,F2>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), total_work_packages, n, shifts, std::ref(func), dim, std::ref(integral_transform), -1, std::ref(time_in_ns_per_thread[thread_number]), std::ref(points_computed_per_thread[thread_number]) ) ); // Launch cpu workers
                        thread_id += 1;
                        thread_number += 1;
                    }
                }
                // Destroy threadpool
                for( std::thread& thread : thread_pool )
                    thread.join();
                thread_pool.clear();

                if(verbosity > 2)
                {
                    for( U i=0; i< extra_threads; i++)
                    {
                        logger << "(" << i << ") Million Function Evaluations/s: " << D(1000)*D(points_computed_per_thread[i])/D(time_in_ns_per_thread[i]) << " Mfeps (Approx)" << std::endl;
                    }
                    if( devices.count(-1) != 0)
                    {
                        D time_in_ns_on_cpu = 0;
                        U points_computed_on_cpu = 0;
                        for( U i=extra_threads; i< extra_threads+cputhreads; i++)
                        {
                            points_computed_on_cpu += points_computed_per_thread[i];
                            time_in_ns_on_cpu = std::max(time_in_ns_on_cpu,time_in_ns_per_thread[i]);
                        }
                        logger << "(-1) Million Function Evaluations/s: " << D(1000)*D(points_computed_on_cpu)/D(time_in_ns_on_cpu) << " Mfeps (Approx)" << std::endl;
                    }
                }
            }

            std::chrono::steady_clock::time_point time_after_compute = std::chrono::steady_clock::now();

            if(verbosity > 2)
            {
                D mfeps = D(1000)*D(n*shifts)/D(std::chrono::duration_cast<std::chrono::nanoseconds>(time_after_compute - time_before_compute).count()); // million function evaluations per second
                logger << "(Total) Million Function Evaluations/s: " << mfeps << " Mfeps" << std::endl;
            }
            res = integrators::core::reduce(r, n, shifts,  previous_iterations, verbosity, logger);
        }
        return res;
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    void Qmc<T,D,U,G,H>::update(result<T,U>& res, U& n, U& m, U& function_evaluations) const
    {
        using std::pow;

        if (verbosity > 2) logger << "-- qmc::update called --" << std::endl;

        const D MAXIMUM_ERROR_RATIO = static_cast<D>(20);
        const D EXPECTED_SCALING = static_cast<D>(0.8); // assume error scales as n^(-expectedScaling)

        function_evaluations += res.n*res.m; // update count of function_evaluations
        if(verbosity > 1 ) logger << "function_evaluations " << function_evaluations << std::endl;

        D error_ratio = std::min(integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode),MAXIMUM_ERROR_RATIO);
        if (error_ratio < static_cast<D>(1))
        {
            if (verbosity > 2) logger << "error goal reached" << std::endl;
            return;
        }
        U new_m = minm;
        #define QMC_POW_CALL pow(error_ratio,static_cast<D>(1)/EXPECTED_SCALING)
        static_assert(std::is_same<decltype(QMC_POW_CALL),D>::value, "Downcast detected in qmc::update(. Please implement \"D pow(D)\".");
        U new_n = get_next_n(static_cast<U>(static_cast<D>(n)*QMC_POW_CALL));
        #undef QMC_POW_CALL
        if ( new_n <= n or ( error_ratio*error_ratio - static_cast<D>(1) < static_cast<D>(new_n)/static_cast<D>(n)))
        {
            // n did not increase, or increasing m will be faster
            // increase m
            if (verbosity > 2) logger << "n did not increase, or increasing m will be faster, increasing m." << std::endl;
            new_n = n;
            new_m = static_cast<U>(static_cast<D>(m)*error_ratio*error_ratio)+1-m;
            if( verbosity > 2 ) logger << new_m << std::endl;
        }
        if ( maxeval < function_evaluations + new_n*new_m)
        {
            // Decrease n
            if ( verbosity > 2 ) logger << "requested number of function evaluations greater than maxeval, reducing n." << std::endl;
            new_n = get_next_n((maxeval-function_evaluations)/new_m);
        }
        n = new_n;
        m = new_m;
        if(verbosity > 1 ) logger << "updated n m " << n << " " << m << std::endl;
    };

    template <typename T, typename D, typename U, typename G, typename H>
    U Qmc<T,D,U,G,H>::get_next_n(U preferred_n) const
    {
        U n;
        if ( generatingvectors.lower_bound(preferred_n) == generatingvectors.end() )
        {
            n = generatingvectors.rbegin()->first;
            if (verbosity > 0) logger << "Qmc integrator does not have generating vector with n larger than " << std::to_string(preferred_n) << ", using largest generating vector with size " << std::to_string(n) << "." << std::endl;
        } else {
            n = generatingvectors.lower_bound(preferred_n)->first;
        }

        // Check n satisfies requirements of mod_mul implementation
        if ( n >= std::numeric_limits<typename std::make_signed<U>::type>::max() ) throw std::domain_error("Qmc integrator called with n larger than the largest finite value representable with the signed type corresponding to U. Please decrease minn or use a larger unsigned integer type for U.");
        if ( n >= std::pow(std::numeric_limits<D>::radix,std::numeric_limits<D>::digits-1) ) throw std::domain_error("Qmc integrator called with n larger than the largest finite value representable by the mantiassa.");

        return n;
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    template <typename F1, typename F2>
    result<T,U> Qmc<T,D,U,G,H>::integrate(F1& func, const U dim, F2& integral_transform)
    {
        if ( dim < 1 ) throw std::invalid_argument("qmc::integrate called with dim < 1. Check that your integrand depends on at least one variable of integration.");
        if ( minm < 2 ) throw std::domain_error("qmc::integrate called with minm < 2. This algorithm can not be used with less than 2 random shifts. Please increase minm.");
        if ( maxmperpackage < 2 ) throw std::domain_error("qmc::integrate called with maxmperpackage < 2. This algorithm can not be used with less than 2 concurrent random shifts. Please increase maxmperpackage.");
        if ( maxnperpackage == 0 ) throw std::domain_error("qmc::integrate called with maxnperpackage = 0. Please set maxnperpackage to a positive integer.");
        if ( cputhreads < 1 ) throw std::domain_error("qmc::integrate called with cputhreads < 1. Please set cputhreads to a positive integer.");

        if (verbosity > 2) logger << "-- qmc::integrate called --" << std::endl;

        std::vector<result<T,U>> previous_iterations; // keep track of the different interations
        U function_evaluations = 0;
        U n = get_next_n(minn); // get next available n >= minn
        U m = minm;
        result<T,U> res;
        do
        {
            if (verbosity > 1) logger << "iterating" << std::endl;
            res = sample(func,dim,integral_transform,n,m, previous_iterations);
            if (verbosity > 1) logger << "result " << res.integral << " " << res.error << std::endl;
            update(res,n,m,function_evaluations);
        } while  ( integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode) > static_cast<D>(1) && function_evaluations < maxeval );
        return res;
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    template <typename F1>
    result<T,U> Qmc<T,D,U,G,H>::integrate(F1& func, const U dim)
    {
        integrators::transforms::Korobov<D,U,3> default_integral_transform;
        return integrate(func, dim, default_integral_transform);
    };
    
    template <typename T, typename D, typename U, typename G, typename H>
    Qmc<T,D,U,G,H>::Qmc() :
    logger(std::cout), randomgenerator( G( std::random_device{}() ) ), minn(8191), minm(32), epsrel(0.01), epsabs(1e-7), maxeval(1000000), maxnperpackage(1), maxmperpackage(1024), errormode(integrators::ErrorMode::all), cputhreads(std::thread::hardware_concurrency()), cudablocks(1024), cudathreadsperblock(256), devices({-1}), generatingvectors(integrators::generatingvectors::cbcpt_dn1_100<U>()), verbosity(0)
    {
        // Check U satisfies requirements of mod_mul implementation
        static_assert( std::numeric_limits<U>::is_modulo, "Qmc integrator constructed with a type U that is not modulo. Please use a different unsigned integer type for U.");
        static_assert( std::numeric_limits<D>::radix == 2, "Qmc integrator constructed with a type D that does not have radix == 2. Please use a different floating point type for D.");
        
        if ( cputhreads == 0 )
        {
            cputhreads = 1; // Correct cputhreads if hardware_concurrency is 0, i.e. not well defined or not computable
            if (verbosity > 1) logger << "Qmc increased cputhreads from 0 to 1." << std::endl;
        }

#ifdef __CUDACC__
        // Get available gpus and add to devices
        int device_count = integrators::core::cuda::get_device_count();
        for(int i = 0; i < device_count; i++)
            devices.insert(i);
#endif
    };
};

#endif
