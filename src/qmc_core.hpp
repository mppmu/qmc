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

namespace integrators
{
    
    template <typename T, typename D, typename U, typename G>
    U Qmc<T,D,U,G>::get_next_n(U preferred_n) const
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
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::init_z(std::vector<U>& z, const U n, const U dim) const
    {
        z = generatingvectors.at(n);
        if ( dim > z.size() ) throw std::domain_error("dim > generating vector dimension. Please supply a generating vector table with a larger number of dimensions.");
        z.resize(dim);
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::init_d(std::vector<D>& d, const U m, const U dim)
    {
        d.clear();
        for (U k = 0; k < m; k++)
            for (U sDim = 0; sDim < dim; sDim++)
                d.push_back(uniformDistribution(randomgenerator));
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::init_r(std::vector<T>& r, const U m, const U r_size) const
    {
        r.clear();
        r.resize(m * r_size, {0.});
    };
    
    template <typename T, typename D, typename U, typename G>
    result<T,U> Qmc<T,D,U,G>::reduce(const std::vector<T>& r, const U n, const U m, std::vector<result<T,U>> & previous_iterations) const
    {
        if (verbosity > 1)
        {
            logger << "-- qmc::reduce called --" << std::endl;
            for(const auto& previous_result : previous_iterations)
            {
                logger << "previous_result: integral " << previous_result.integral << ", error " << previous_result.error << ", n " << previous_result.n << ", m " << previous_result.m << std::endl;
            }
        }

        T mean = {0.};
        T variance = {0.};
        U previous_m = 0;
        if(!previous_iterations.empty())
        {
            result<T,U> & previous_res = previous_iterations.back();
            if(previous_res.n == n)
            {
                if (verbosity>2) logger << "using additional shifts to improve previous iteration" << std::endl;
                previous_m = previous_res.m;
                mean = previous_res.integral*static_cast<T>(n);
                variance = compute_variance_from_error(previous_res.error);
                variance *= static_cast<T>(previous_res.m-1) * static_cast<T>(previous_res.m) * static_cast<T>(previous_res.n) * static_cast<T>(previous_res.n);
                previous_iterations.pop_back();
            }
        }
        for(U k = 0; k < m; k++)
        {
            T sum = {0.};
            T delta = {0.};
            T kahan_c = {0.};
            for (U i = 0; i<r.size()/m; i++)
            {
                // Compute sum using Kahan summation
                // equivalent to: sum += r.at(k*r.size()/m+i);
                T kahan_y = r.at(k*r.size()/m+i) - kahan_c;
                T kahan_t = sum + kahan_y;
                T kahan_d = kahan_t - sum;
                kahan_c = kahan_d - kahan_y;
                sum = kahan_t;
            }
            if (verbosity > 1) logger << "shift " << k+previous_m << " result: " << sum/static_cast<T>(n) << std::endl;
            // Compute Variance using online algorithm (Knuth, The Art of Computer Programming)
            delta = sum - mean;
            mean = mean + delta/(static_cast<T>(k+previous_m+1));
            variance = compute_variance(mean, variance, sum, delta);
        }
        T integral = mean/(static_cast<T>(n));
        variance = variance/( static_cast<T>(m+previous_m-1) * static_cast<T>(m+previous_m) * static_cast<T>(n) * static_cast<T>(n) ); // variance of the mean
        T error = compute_error(variance);
        previous_iterations.push_back({integral, error, n, m+previous_m});
        if (verbosity > 0)
            logger << "integral " << integral << ", error " << error << ", n " << n << ", m " << m+previous_m << std::endl;
        return {integral, error, n, m+previous_m};
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    void Qmc<T,D,U,G>::compute(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size, const U total_work_packages, const U n, const U m, F1& func, const U dim, F2& integral_transform) const
    {
        for (U k = 0; k < m; k++)
        {
            T kahan_c = {0.};
            for( U offset = i; offset < n; offset += total_work_packages )
            {
                D wgt = 1.;
                D mynull = 0;
                std::vector<D> x(dim,0);

                for (U sDim = 0; sDim < dim; sDim++)
                {
                    x[sDim] = std::modf( integrators::mul_mod<D,D,U>(offset,z.at(sDim),n)/(static_cast<D>(n)) + d.at(k*dim+sDim), &mynull);
                }

                integral_transform(x.data(), wgt, dim);

                // Nudge point inside border (for numerical stability)
                for (U sDim = 0; sDim < dim; sDim++)
                {
                    if( x[sDim] < border)
                    x[sDim] = border;
                    if( x[sDim] > 1.-border)
                    x[sDim] = 1.-border;
                }

                T point = func(x.data());

                // Compute sum using Kahan summation
                // equivalent to: r_element[k*r_size] += wgt*point;
                T kahan_y = wgt*point - kahan_c;
                T kahan_t = r_element[k*r_size] + kahan_y;
                T kahan_d = kahan_t - r_element[k*r_size];
                kahan_c = kahan_d - kahan_y;
                r_element[k*r_size] = kahan_t;
            }
        }
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    void Qmc<T,D,U,G>::compute_worker(const U thread_id, U& work_queue, std::mutex& work_queue_mutex, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U n, const U m, F1& func, const U dim, F2& integral_transform, const int device) const
    {
#ifdef __CUDACC__
        // define device pointers (must be accessible in local scope of the entire function)
        std::unique_ptr<integrators::detail::cuda_memory<F1>> d_func;
        std::unique_ptr<integrators::detail::cuda_memory<F2>> d_integral_transform;
#endif

        U i;
        U  work_this_iteration;
        if (device == -1) {
            work_this_iteration = 1;
        } else {
            work_this_iteration = cudablocks*cudathreadsperblock;
#ifdef __CUDACC__
            setup_gpu(d_func, func, d_integral_transform, integral_transform, device, verbosity, logger);
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
            }
            else if (work_queue >= work_this_iteration)
            {
                work_queue-=work_this_iteration;
                i = work_queue;
            }
            else
            {
                work_this_iteration = work_queue; // TODO - do not redo work in this case...
                work_queue = 0;
                i = 0;
            }
            work_queue_mutex.unlock();
            
            if( !work_remaining )
                break;
            
            // Do work
            if (device == -1)
            {
                compute(i, z, d, &r[thread_id], r.size()/m, total_work_packages, n, m, func, dim, integral_transform);
            }
            else
            {
#ifdef __CUDACC__
                compute_gpu(i, z, d, &r[thread_id], r.size()/m, work_this_iteration, total_work_packages, n, m,
                            static_cast<typename std::remove_const<F1>::type*>(*d_func), dim,
                            static_cast<typename std::remove_const<F2>::type*>(*d_integral_transform), device, cudablocks, cudathreadsperblock);
#endif
            }
        }
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    result<T,U> Qmc<T,D,U,G>::sample(F1& func, const U dim, F2& integral_transform, const U n, const U m, std::vector<result<T,U>> & previous_iterations)
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
        U r_size = extra_threads*cudablocks*cudathreadsperblock; // non-cpu workers
        if (devices.count(-1) != 0)
        {
            r_size += cputhreads; // cpu-workers
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
            init_r(r, shifts, r_size);

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
                logger << "r " << shifts << "*" << r_size << std::endl;
            }

            if ( cputhreads == 1 && devices.size() == 1 && devices.count(-1) == 1)
            {
                // Compute serially on cpu
                if (verbosity > 2) logger << "computing serially" << std::endl;
                for( U i=0; i < total_work_packages; i++)
                {
                    compute(i, z, d, &r[0], r.size()/shifts, total_work_packages, n, shifts, func, dim, integral_transform);
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
                std::vector<std::thread> thread_pool;
                thread_pool.reserve(cputhreads+extra_threads);
                for (int device : devices)
                {
                    if( device != -1)
                    {
#ifdef __CUDACC__
                        thread_pool.push_back( std::thread( &Qmc<T,D,U,G>::compute_worker<F1,F2>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), total_work_packages, n, shifts, std::ref(func), dim, std::ref(integral_transform), device ) ); // Launch non-cpu workers
                        thread_id += cudablocks*cudathreadsperblock;
#else
                        throw std::invalid_argument("qmc::sample called with device != -1 (CPU) but CUDA not supported by compiler, device: " + std::to_string(device));
#endif
                    }
                }
                if( devices.count(-1) != 0)
                {
                    for ( U i=0; i < cputhreads; i++)
                    {
                        thread_pool.push_back( std::thread( &Qmc<T,D,U,G>::compute_worker<F1,F2>, this, thread_id, std::ref(work_queue), std::ref(work_queue_mutex), std::cref(z), std::cref(d), std::ref(r), total_work_packages, n, shifts, std::ref(func), dim, std::ref(integral_transform), -1 ) ); // Launch cpu workers
                        thread_id += 1;
                    }
                }
                // Destroy threadpool
                for( std::thread& thread : thread_pool )
                    thread.join();
                thread_pool.clear();
            }
            res = reduce(r, n, shifts,  previous_iterations);
        }
        return res;
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::update(result<T,U>& res, U& n, U& m) const
    {
        if (verbosity > 2) logger << "-- qmc::update called --" << std::endl;

        const D MAXIMUM_ERROR_RATIO = static_cast<D>(20);
        const D EXPECTED_SCALING = static_cast<D>(0.8); // assume error scales as n^(-expectedScaling)

        D error_ratio = std::min(compute_error_ratio(res, epsrel, epsabs, errormode),MAXIMUM_ERROR_RATIO);
        if (error_ratio < static_cast<D>(1))
        {
            if (verbosity > 2) logger << "error goal reached" << std::endl;
            return;
        }
        U new_m = minm;
        U new_n = get_next_n(static_cast<U>(static_cast<D>(n)*std::pow(error_ratio,static_cast<D>(1)/EXPECTED_SCALING)));
        if ( new_n <= n or ( error_ratio*error_ratio - static_cast<D>(1) < static_cast<D>(new_n)/static_cast<D>(n)))
        {
            // n did not increase, or increasing m will be faster
            // increase m
            new_n = n;
            new_m = static_cast<U>(static_cast<D>(m)*error_ratio*error_ratio)+1-m;
        }
        if ( maxeval < new_n*new_m)
        {
            // Decrease n
            new_n = get_next_n(maxeval/new_m);
        }
        n = new_n;
        m = new_m;
        if(verbosity > 1 ) logger << "updated n m " << n << " " << m << std::endl;
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    result<T,U> Qmc<T,D,U,G>::integrate(F1& func, const U dim, F2& integral_transform)
    {
        if ( dim < 1 ) throw std::invalid_argument("qmc::integrate called with dim < 1. Check that your integrand depends on at least one variable of integration.");
        if ( minm < 2 ) throw std::domain_error("qmc::integrate called with minm < 2. This algorithm can not be used with less than 2 random shifts. Please increase minm.");
        if ( maxmperpackage < 2 ) throw std::domain_error("qmc::integrate called with maxmperpackage < 2. This algorithm can not be used with less than 2 concurrent random shifts. Please increase maxmperpackage.");
        if ( maxnperpackage == 0 ) throw std::domain_error("qmc::integrate called with maxnperpackage = 0. Please set maxnperpackage to a positive integer.");

        if (verbosity > 2) logger << "-- qmc::integrate called --" << std::endl;

        std::vector<result<T,U>> previous_iterations; // keep track of the different interations

        U n = get_next_n(minn); // get next available n >= minn
        U m = minm;
        if ( maxeval < minn*minm)
        {
            if (verbosity > 2) logger << "increasing maxeval " << maxeval << " -> " << minn*minm << std::endl;
            maxeval = minn*minm;
        }
        result<T,U> res;
        do
        {
            if (verbosity > 1) logger << "iterating" << std::endl;
            res = sample(func,dim,integral_transform,n,m, previous_iterations);
            if (verbosity > 1) logger << "result " << res.integral << " " << res.error << std::endl;
            update(res,n,m);
        } while  ( compute_error_ratio(res, epsrel, epsabs, errormode) > static_cast<D>(1) && (res.n*res.m) < maxeval ); // TODO - if error estimate is not decreasing quit
        return res;
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1>
    result<T,U> Qmc<T,D,U,G>::integrate(F1& func, const U dim)
    {
        integrators::Korobov3<D,U> default_integral_transform;
        return integrate(func, dim, default_integral_transform);
    };
    
    template <typename T, typename D, typename U, typename G>
    Qmc<T,D,U,G>::Qmc() :
    logger(std::cout), randomgenerator( G( std::random_device{}() ) ), minn(8191), minm(32), epsrel(std::numeric_limits<D>::max()), epsabs(std::numeric_limits<D>::max()), border(0), maxeval(std::numeric_limits<U>::max()), maxnperpackage(1), maxmperpackage(1024), errormode(all), cputhreads(std::thread::hardware_concurrency()), cudablocks(1024), cudathreadsperblock(256), devices({-1}), verbosity(0)
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
        int device_count = get_device_count_gpu();
        for(int i = 0; i < device_count; i++)
            devices.insert(i);
#endif
        
        init_g();
    };
    
};
