#ifndef QMC_MEMBERS_H
#define QMC_MEMBERS_H

#include <cstddef> // size_t
#include <cmath> // modf, abs, sqrt, pow
#include <stdexcept> // domain_error, invalid_argument
#include <thread> // thread
#include <algorithm> // min, max
#include <type_traits> // make_signed, enable_if
#include <limits> // numeric_limits
#include <string> // to_string
#include <vector>
#include <iostream>
#include <iterator> // advance
#include <mutex>
#include <memory> // unique_ptr
#include <numeric> // partial_sum
#include <cassert> // assert
#include <chrono>

#include <gsl/gsl_multifit_nlinear.h>

namespace integrators
{
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    void Qmc<T,D,M,P,F,G,H>::init_z(std::vector<U>& z, const U n, const U number_of_integration_variables) const
    {
        z = generatingvectors.at(n);
        if ( number_of_integration_variables > z.size() )
            throw std::domain_error("number_of_integration_variables > generating vector dimension. Please supply a generating vector table with a larger number of dimensions.");
        z.resize(number_of_integration_variables);
    };
    
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    void Qmc<T,D,M,P,F,G,H>::init_d(std::vector<D>& d, const U m, const U number_of_integration_variables)
    {
        d.clear();
        d.reserve(m*number_of_integration_variables);
        for (U k = 0; k < m; k++)
            for (U sDim = 0; sDim < number_of_integration_variables; sDim++)
                d.push_back(uniform_distribution(randomgenerator));
    };
    
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    void Qmc<T,D,M,P,F,G,H>::init_r(std::vector<T>& r, const U m, const U r_size_over_m) const
    {
        r.clear();
        r.resize(m * r_size_over_m, {0.});
    };
    
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    void Qmc<T,D,M,P,F,G,H>::sample_worker(const U thread_id, U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m, I& func, const int device, D& time_in_ns, U& points_computed) const
    {
        std::chrono::steady_clock::time_point time_before_compute = std::chrono::steady_clock::now();

        points_computed = 0;

        // Setup worker
#ifdef __CUDACC__
        // define device pointers (must be accessible in local scope of the entire function)
        U d_r_size = m*cudablocks*cudathreadsperblock;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<I>> d_func;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>> d_z;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>> d_d;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>> d_r;
#endif
        U i;
        U  work_this_iteration;
        if (device == -1) {
            work_this_iteration = 1;
        } else {
#ifdef __CUDACC__
            work_this_iteration = cudablocks*cudathreadsperblock;
            integrators::core::cuda::setup_sample(d_z, z, d_d, d, d_r, d_r_size/m, &r[thread_id], r.size()/m, m, d_func, func, device, verbosity, logger);
#else
            throw std::invalid_argument("qmc::sample called with device != -1 (CPU) but CUDA not supported by compiler, device: " + std::to_string(device));
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
                integrators::core::generic::compute(i, z, d, &r[thread_id], r.size()/m, total_work_packages, n, m, func);
            }
            else
            {
#ifdef __CUDACC__
                integrators::core::cuda::compute<M>(*this, i, work_this_iteration, total_work_packages, d_z, d_d, d_r, d_r_size/m, n, m, d_func, device);
#endif
            }

            points_computed += work_this_iteration*m;

        }

        // Teardown worker
#ifdef __CUDACC__
        if (device != -1) {
            integrators::core::cuda::teardown_sample(d_r, d_r_size/m, &r[thread_id], r.size()/m, m, device, verbosity, logger);
        }
#endif

        std::chrono::steady_clock::time_point time_after_compute = std::chrono::steady_clock::now();
        time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_after_compute - time_before_compute).count();

    };
    
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    result<T> Qmc<T,D,M,P,F,G,H>::sample(I& func, const U n, const U m, std::vector<result<T>> & previous_iterations)
    {
        std::vector<U> z;
        std::vector<D> d;
        std::vector<T> r;

        result<T> res;

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
            init_z(z, n, func.number_of_integration_variables);
            init_d(d, shifts, func.number_of_integration_variables);
            init_r(r, shifts, r_size_over_m);

            if (verbosity > 0)
            {
                logger << "-- qmc::sample called --" << std::endl;
                logger << "func.number_of_integration_variables " << func.number_of_integration_variables << std::endl;
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
                bool display_timing = logger.display_timing;
                logger.display_timing = false;
                for (const int& i : devices)
                    logger << i << " ";
                logger << std::endl;
                logger.display_timing = display_timing;
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
                    integrators::core::generic::compute(i, z, d, &r[0], r.size()/shifts, total_work_packages, n, shifts, func);
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
                        thread_pool.push_back( std::thread( &Qmc<T,D,M,P,F,G,H>::sample_worker<I>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), total_work_packages, n, shifts, std::ref(func), device, std::ref(time_in_ns_per_thread[thread_number]), std::ref(points_computed_per_thread[thread_number])  ) ); // Launch non-cpu workers
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
                        thread_pool.push_back( std::thread( &Qmc<T,D,M,P,F,G,H>::sample_worker<I>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), total_work_packages, n, shifts, std::ref(func), -1, std::ref(time_in_ns_per_thread[thread_number]), std::ref(points_computed_per_thread[thread_number]) ) ); // Launch cpu workers
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
    
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    void Qmc<T,D,M,P,F,G,H>::evaluate_worker(const U thread_id, U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U n, I& func, const int device, D& time_in_ns, U& points_computed) const
    {
        std::chrono::steady_clock::time_point time_before_compute = std::chrono::steady_clock::now();

        points_computed = 0;

        // Setup worker
#ifdef __CUDACC__
        // define device pointers (must be accessible in local scope of the entire function)
        U d_r_size = cudablocks*cudathreadsperblock;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<I>> d_func;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>> d_z;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>> d_d;
        std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>> d_r;
#endif
        U i;
        U  work_this_iteration;
        if (device == -1) {
            work_this_iteration = 1;
        } else {
#ifdef __CUDACC__
            work_this_iteration = cudablocks*cudathreadsperblock;
            integrators::core::cuda::setup_evaluate(d_z, z, d_d, d, d_r, d_r_size, d_func, func, device, verbosity, logger);
#else
            throw std::invalid_argument("qmc::sample called with device != -1 (CPU) but CUDA not supported by compiler, device: " + std::to_string(device));
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
                integrators::core::generic::generate_samples(i, z, d, &r[i], n, func);
            }
            else
            {
#ifdef __CUDACC__
                integrators::core::cuda::generate_samples<M>(*this, i, work_this_iteration, d_z, d_d, d_r, n, d_func, device);
#endif
            }

            points_computed += work_this_iteration;


            // copy results to host
#ifdef __CUDACC__
            if (device != -1) {
                integrators::core::cuda::copy_back(d_r, work_this_iteration, &r[i], device, verbosity, logger);
            }
#endif
        }

        std::chrono::steady_clock::time_point time_after_compute = std::chrono::steady_clock::now();
        time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_after_compute - time_before_compute).count();

    };
    
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    samples<T,D> Qmc<T,D,M,P,F,G,H>::evaluate(I& func)
    {
        if ( func.number_of_integration_variables < 1 )
            throw std::invalid_argument("qmc::evaluate called with func.number_of_integration_variables < 1. Check that your integrand depends on at least one variable of integration.");
        if ( func.number_of_integration_variables > M )
            throw std::invalid_argument("qmc::evaluate called with func.number_of_integration_variables > M. Please increase M (maximal number of integration variables).");
        if ( cputhreads < 1 )
            throw std::domain_error("qmc::evaluate called with cputhreads < 1. Please set cputhreads to a positive integer.");
#ifndef __CUDACC__
        if ( devices.size() != 1 || devices.count(-1) != 1)
            throw std::invalid_argument("qmc::evaluate called with devices != {-1} (CPU) but CUDA not supported by compiler.");
#endif

        // allocate memory
        samples<T,D> res;
        U& n = res.n;
        n = get_next_n(evaluateminn); // get next available n >= evaluateminn
        std::vector<U>& z = res.z;
        std::vector<D>& d = res.d;
        std::vector<T>& r = res.r;

        // initialize z, d, r
        init_z(z, n, func.number_of_integration_variables);
        init_d(d, 1, func.number_of_integration_variables);
        init_r(r, 1, n); // memory required for result vector

        U extra_threads = devices.size() - devices.count(-1);

        if (verbosity > 0)
        {
            logger << "-- qmc::evaluate called --" << std::endl;
            logger << "func.number_of_integration_variables " << func.number_of_integration_variables << std::endl;
            logger << "evaluateminn " << evaluateminn << std::endl;
            logger << "cputhreads " << cputhreads << std::endl;
            logger << "cudablocks " << cudablocks << std::endl;
            logger << "cudathreadsperblock " << cudathreadsperblock << std::endl;
            logger << "devices ";
            bool display_timing = logger.display_timing;
            logger.display_timing = false;
            for (const int& i : devices)
                logger << i << " ";
            logger << std::endl;
            logger.display_timing = display_timing;
            logger << "n " << n << std::endl;
        }

        std::chrono::steady_clock::time_point time_before_compute = std::chrono::steady_clock::now();

        if ( cputhreads == 1 && devices.size() == 1 && devices.count(-1) == 1)
        {
            // Compute serially on cpu
            if (verbosity > 2) logger << "computing serially" << std::endl;
            for( U i=0; i < n; i++)
            {
                integrators::core::generic::generate_samples(i, z, d, &r[i], n, func);
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
            U work_queue = n;

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
                    thread_pool.push_back( std::thread( &Qmc<T,D,M,P,F,G,H>::evaluate_worker<I>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), n, std::ref(func), device, std::ref(time_in_ns_per_thread[thread_number]), std::ref(points_computed_per_thread[thread_number]) ) ); // Launch non-cpu workers
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
                    thread_pool.push_back( std::thread( &Qmc<T,D,M,P,F,G,H>::evaluate_worker<I>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), n, std::ref(func), -1, std::ref(time_in_ns_per_thread[thread_number]), std::ref(points_computed_per_thread[thread_number]) ) ); // Launch cpu workers
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
            D mfeps = D(1000)*D(n)/D(std::chrono::duration_cast<std::chrono::nanoseconds>(time_after_compute - time_before_compute).count()); // million function evaluations per second
            logger << "(Total) Million Function Evaluations/s: " << mfeps << " Mfeps" << std::endl;
        }

        return res;
    };

    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    typename F<I,D,M>::transform_t Qmc<T,D,M,P,F,G,H>::fit(I& func)
    {
        using std::abs;

        typename F<I,D,M>::function_t fit_function;
        typename F<I,D,M>::jacobian_t fit_function_jacobian;
        typename F<I,D,M>::hessian_t fit_function_hessian;
        typename F<I,D,M>::transform_t fit_function_transform(func);

        if (fit_function.num_parameters <= 0)
            return fit_function_transform;

        std::vector<D> x,y;
        std::vector<std::vector<D>> fit_parameters;
        fit_parameters.reserve(func.number_of_integration_variables);

        // Generate data to be fitted
        integrators::samples<T,D> result = evaluate(func);

        // fit fit_function
        for (U sdim = 0; sdim < func.number_of_integration_variables; ++sdim)
        {
            // compute the x values
            std::vector<D> unordered_x;
            unordered_x.reserve(result.n);
            for (size_t i = 0; i < result.n; ++i)
            {
                unordered_x.push_back( result.get_x(i, sdim) );
            }

            // sort by x value
            std::vector<size_t> sort_key = math::argsort(unordered_x);
            x.clear();
            y.clear();
            x.reserve( sort_key.size() );
            y.reserve( sort_key.size() );
            for (const auto& idx : sort_key)
            {
                x.push_back( unordered_x.at(idx) );
                y.push_back( abs(result.r.at(idx)) );
            }

            // compute cumulative sum
            std::partial_sum(y.begin(), y.end(), y.begin());
            for (auto& element : y)
            {
                element /= y.back();
            }

            // reduce number of sampling points for fit
            std::vector<D> xx;
            std::vector<D> yy;
            for ( size_t i = fitstepsize/2; i<x.size(); i+=fitstepsize)
            {
                    xx.push_back(x.at(i));
                    yy.push_back(y.at(i));
            }

            // run a least squares fit
            fit_parameters.push_back( core::least_squares(fit_function,fit_function_jacobian, fit_function_hessian, yy,xx,verbosity,logger, fitmaxiter, fitxtol, fitgtol, fitftol, fitparametersgsl) );
        }

        for (size_t d = 0; d < fit_function_transform.number_of_integration_variables; ++d)
            for (size_t i = 0; i < fit_parameters.at(d).size(); ++i)
                fit_function_transform.p[d][i] = fit_parameters.at(d).at(i);

        return fit_function_transform;
    };

    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    void Qmc<T,D,M,P,F,G,H>::update(const result<T>& res, U& n, U& m) const
    {
        using std::pow;

        if (verbosity > 2) logger << "-- qmc::update called --" << std::endl;

        const D MAXIMUM_ERROR_RATIO = static_cast<D>(20);
        const D EXPECTED_SCALING = static_cast<D>(0.8); // assume error scales as n^(-expectedScaling)

        D error_ratio = std::min(integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode),MAXIMUM_ERROR_RATIO);
        if (error_ratio < static_cast<D>(1))
        {
            if (verbosity > 2) logger << "error goal reached" << std::endl;
            return;
        }

        if(res.evaluations > maxeval)
        {
            if (verbosity > 2) logger << "maxeval reached" << std::endl;
            return;
        }

        U new_m = minm;
        #define QMC_POW_CALL pow(error_ratio,static_cast<D>(1)/EXPECTED_SCALING)
        static_assert(std::is_same<decltype(QMC_POW_CALL),D>::value, "Downcast detected in qmc::update(. Please implement \"D pow(D)\".");
        U new_n = get_next_n(static_cast<U>(static_cast<D>(n)*QMC_POW_CALL));
        #undef QMC_POW_CALL
        if ( new_n <= n or ( error_ratio*error_ratio - static_cast<D>(1) < static_cast<D>(new_n)/static_cast<D>(n)))
        {
            // n did not increase, or increasing m will be faster, increase m
            new_n = n;
            new_m = static_cast<U>(static_cast<D>(m)*error_ratio*error_ratio)+1-m;
            if (verbosity > 2)
                logger << "n did not increase, or increasing m will be faster, increasing m to " << new_m << "." << std::endl;
        }
        if ( maxeval < res.evaluations + new_n*new_m)
        {
            // Decrease new_n
            if ( verbosity > 2 )
                logger << "requested number of function evaluations greater than maxeval, reducing n." << std::endl;
            new_n = std::max(n, get_next_n((maxeval-res.evaluations)/new_m));
        }
        if ( n == new_n && maxeval < res.evaluations + new_n*new_m)
        {
            // Decrease new_m
            if ( verbosity > 2 )
                logger << "requested number of function evaluations greater than maxeval, reducing m." << std::endl;
            new_m = std::max(U(1),(maxeval-res.evaluations)/new_n);
        }
        n = new_n;
        m = new_m;
        if(verbosity > 1 ) logger << "updated n m " << n << " " << m << std::endl;

    };

    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    U Qmc<T,D,M,P,F,G,H>::get_next_n(U preferred_n) const
    {
        U n;
        if ( generatingvectors.lower_bound(preferred_n) == generatingvectors.end() )
        {
            n = generatingvectors.rbegin()->first;
            if (verbosity > 0)
                logger << "Qmc integrator does not have a generating vector with n larger than " << std::to_string(preferred_n) << ", using largest generating vector with size " << std::to_string(n) << "." << std::endl;
        } else {
            n = generatingvectors.lower_bound(preferred_n)->first;
        }

        // Check n satisfies requirements of mod_mul implementation
        if ( n >= std::numeric_limits<typename std::make_signed<U>::type>::max() )
            throw std::domain_error("Qmc integrator called with n larger than the largest finite value representable with the signed type corresponding to U. Please decrease minn or use a larger unsigned integer type for U.");
        if ( n >= std::pow(std::numeric_limits<D>::radix,std::numeric_limits<D>::digits-1) )
            throw std::domain_error("Qmc integrator called with n larger than the largest finite value representable by the mantissa.");

        return n;
    };

    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    result<T> Qmc<T,D,M,P,F,G,H>::integrate_no_fit_no_transform(I& func)
    {
        if ( func.number_of_integration_variables < 1 )
            throw std::invalid_argument("qmc::integrate called with func.number_of_integration_variables < 1. Check that your integrand depends on at least one variable of integration.");
        if ( func.number_of_integration_variables > M )
            throw std::invalid_argument("qmc::integrate called with func.number_of_integration_variables > M. Please increase M (maximal number of integration variables).");
        if ( minm < 2 )
            throw std::domain_error("qmc::integrate called with minm < 2. This algorithm can not be used with less than 2 random shifts. Please increase minm.");
        if ( maxmperpackage < 2 )
            throw std::domain_error("qmc::integrate called with maxmperpackage < 2. This algorithm can not be used with less than 2 concurrent random shifts. Please increase maxmperpackage.");
        if ( maxnperpackage == 0 )
            throw std::domain_error("qmc::integrate called with maxnperpackage = 0. Please set maxnperpackage to a positive integer.");
        if ( cputhreads < 1 )
            throw std::domain_error("qmc::integrate called with cputhreads < 1. Please set cputhreads to a positive integer.");
#ifndef __CUDACC__
        if ( devices.size() != 1 || devices.count(-1) != 1)
            throw std::invalid_argument("qmc::integrate called with devices != {-1} (CPU) but CUDA not supported by compiler.");
#endif

        if (verbosity > 2)
            logger << "-- qmc::integrate called --" << std::endl;

        std::vector<result<T>> previous_iterations; // keep track of the different interations
        U n = get_next_n(minn); // get next available n >= minn
        U m = minm;
        result<T> res;
        do
        {
            if (verbosity > 1)
                logger << "iterating" << std::endl;
            res = sample(func,n,m, previous_iterations);
            if (verbosity > 1)
                logger << "result " << res.integral << " " << res.error << std::endl;
            update(res,n,m);
        } while  ( integrators::overloads::compute_error_ratio(res, epsrel, epsabs, errormode) > static_cast<D>(1) && res.evaluations < maxeval );
        if (verbosity > 0)
        {
            logger << "-- qmc::integrate returning result --" << std::endl;
            logger << "integral " << res.integral << std::endl;
            logger << "error " << res.error << std::endl;
            logger << "n " << res.n << std::endl;
            logger << "m " << res.m << std::endl;
            logger << "iterations " << res.iterations << std::endl;
            logger << "evaluations " << res.evaluations << std::endl;

        }
        return res;
    };

    /*
     * implementation of integrate
     */
    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    template <typename I>
    result<T> Qmc<T,D,M,P,F,G,H>::integrate(I& func)
    {
        typename F<I,D,M>::transform_t fitted_func = fit(func);
        P<typename F<I,D,M>::transform_t,D,M> transformed_fitted_func(fitted_func);
        return integrate_no_fit_no_transform(transformed_fitted_func);
    };

    template <typename T, typename D, U M, template<typename,typename,U> class P, template<typename,typename,U> class F, typename G, typename H>
    Qmc<T,D,M,P,F,G,H>::Qmc() :
    logger(std::cout), randomgenerator( G( std::random_device{}() ) ), minn(8191), minm(32), epsrel(0.01), epsabs(1e-7), maxeval(1000000), maxnperpackage(1), maxmperpackage(1024), errormode(integrators::ErrorMode::all), cputhreads(std::thread::hardware_concurrency()), cudablocks(1024), cudathreadsperblock(256), devices({-1}), generatingvectors(integrators::generatingvectors::cbcpt_dn1_100()), verbosity(0), evaluateminn(100000), fitstepsize(10), fitmaxiter(40), fitxtol(3e-3), fitgtol(1e-8), fitftol(1e-8), fitparametersgsl({})
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

        fitparametersgsl = gsl_multifit_nlinear_default_parameters();
        fitparametersgsl.trs = gsl_multifit_nlinear_trs_lmaccel;
    };
};

#endif
