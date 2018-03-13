#include <cstddef> // size_t
#include <cmath> // modf, abs, sqrt, pow
#include <stdexcept> // domain_error, invalid_argument
#include <thread> // thread
#include <algorithm> // min, max
#include <type_traits> // make_signed
#include <limits> // numeric_limits
#include <string>
#include <vector>
#include <iostream>
#include <iterator> // advance
namespace integrators
{
    
    template <typename T, typename D, typename U, typename G>
    U Qmc<T,D,U,G>::getNextN(U preferred_n) const
    {
        U n;
        if ( generatingVectors.lower_bound(preferred_n) == generatingVectors.end() )
        {
            n = generatingVectors.rbegin()->first;
            if (verbosity > 0) std::cout << "Qmc integrator does not have generating vector with n larger than " << std::to_string(preferred_n) << ", using largest generating vector with size " << std::to_string(n) << "." << std::endl;
        } else {
            n = generatingVectors.lower_bound(preferred_n)->first;
        }
        
        // Check n satisfies requirements of mod_mul implementation
        if ( n >= std::numeric_limits<typename std::make_signed<U>::type>::max() ) throw std::domain_error("Qmc integrator called with n larger than the largest finite value representable with the signed type corresponding to U. Please decrease minn or use a larger unsigned integer type for U.");
        if ( n >= std::pow(std::numeric_limits<D>::radix,std::numeric_limits<D>::digits-1) ) throw std::domain_error("Qmc integrator called with n larger than the largest finite value representable by the mantiassa ");
        
        return n;
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initz(std::vector<U>& z, const U n, const U dim) const
    {
        z = generatingVectors.at(n);
        if ( dim > z.size() ) throw std::domain_error("dim > generating vector dimension. Please supply a generating vector table with a larger number of dimensions");
        z.resize(dim);
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initd(std::vector<D>& d, const U m, const U dim)
    {
        d.clear();
        for (U k = 0; k < m; k++)
            for (U sDim = 0; sDim < dim; sDim++)
                d.push_back(uniformDistribution(randomGenerator));
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initr(std::vector<T>& r, const U m, const U r_size) const
    {
        r.clear();
        r.resize(m * r_size, {0.});
    };
    
    template <typename T, typename D, typename U, typename G>
    result<T,U> Qmc<T,D,U,G>::reduce(const std::vector<T>& r, const U n, const U m) const
    {
        T mean = {0.};
        T variance = {0.};
        for(U k = 0; k < m; k++)
        {
            T sum = {0.};
            T delta = {0.};
            T kahanC = {0.};
            for (U i = 0; i<r.size()/m; i++)
            {
                // Compute sum using Kahan summation
                // equivalent to: sum += r.at(k*r.size()/m+i);
                T kahanY = r.at(k*r.size()/m+i) - kahanC;
                T kahanT = sum + kahanY;
                T kahanD = kahanT - sum;
                kahanC = kahanD - kahanY;
                sum = kahanT;
            }
            if (verbosity > 2) std::cout << "shift " << k << " result: " << sum/static_cast<T>(n) << std::endl;
            // Compute Variance using online algorithm (Knuth, The Art of Computer Programming)
            delta = sum - mean;
            mean = mean + delta/(static_cast<T>(k+1));
            variance = computeVariance(mean, variance, sum, delta);
        }
        T integral = mean/(static_cast<T>(n));
        variance = variance/( static_cast<T>(m-1) * static_cast<T>(m) * static_cast<T>(n) * static_cast<T>(n) ); // variance of the mean
        T error = computeError(variance);
        return {integral, error, n, m};
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    void Qmc<T,D,U,G>::compute(const int i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size, const U total_work_packages, const U points_per_package, const U n, const U m, F1& func, const U dim, F2& integralTransform)
    {
        for (U k = 0; k < m; k++)
        {
            T kahanC = {0.};
            for( U b = 0; b < points_per_package; b++ )
            {
                U offset = b * total_work_packages;
                if(offset + i < n)
                {
                    D wgt = 1.;
                    D mynull = 0;
                    std::vector<D> x(dim,0);
                    
                    for (U sDim = 0; sDim < dim; sDim++)
                        x[sDim] = std::modf( integrators::mul_mod<D,D,U>(i+offset,z.at(sDim),n)/(static_cast<D>(n)) + d.at(k*dim+sDim), &mynull);
                    
                    integralTransform(x.data(), wgt, dim);
                    
                    T point = func(x.data());
                    
                    // Compute sum using Kahan summation
                    // equivalent to: r_element[k*r_size] += wgt*point;
                    T kahanY = wgt*point - kahanC;
                    T kahanT = r_element[k*r_size] + kahanY;
                    T kahanD = kahanT - r_element[k*r_size];
                    kahanC = kahanD - kahanY;
                    r_element[k*r_size] = kahanT;
                }
            }
        }
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    void Qmc<T,D,U,G>::compute_worker(const U thread_id, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U total_work_packages, const U points_per_package, const U n, const U m, F1& func, const U dim, F2& integralTransform, const int device)
    {
        if(verbosity > 1) std::cout << "-(" << thread_id << ") Thread started for device" << device << std::endl;
        
        U i;
        U  work_this_iteration;
        if (device == -1)
            work_this_iteration = 1;
        else
            work_this_iteration = cudablocks*cudathreadsperblock;
        
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
                compute(i, z, d, &r[thread_id], r.size()/m, total_work_packages, points_per_package, n, m, func, dim, integralTransform);
            }
            else
            {
#ifdef __CUDACC__
                compute_gpu(i, z, d, &r[thread_id], r.size()/m, work_this_iteration, total_work_packages, points_per_package, n, m, func, dim, integralTransform, device, cudablocks, cudathreadsperblock);
#endif
            }
        }
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    result<T,U> Qmc<T,D,U,G>::sample(F1& func, const U dim, F2& integralTransform, const U n, const U m)
    {
        std::vector<U> z;
        std::vector<D> d;
        std::vector<T> r;
        
        U total_work_packages = std::min(max_work_packages, n); // Set total number of work packages to be computed
        U points_per_package = (n+total_work_packages-1)/total_work_packages; // Points to compute per thread per work_package
        U extra_threads = devices.size() - devices.count(-1);
        
        // Memory required for result vector
        U r_size = extra_threads*cudablocks*cudathreadsperblock; // non-cpu workers
        if (devices.count(-1) != 0)
        {
            r_size += cputhreads; // cpu-workers
        }

        // Generate z, d, r
        initz(z, n, dim);
        initd(d, m, dim);
        initr(r, m, r_size);
        
        if (verbosity > 0)
        {
            std::cout << "-- qmc::sample called --" << std::endl;
            std::cout << "dim " << dim << std::endl;
            std::cout << "minn " << minn << std::endl;
            std::cout << "minm " << minm << std::endl;
            std::cout << "epsrel " << epsrel << std::endl;
            std::cout << "epsabs " << epsabs << std::endl;
            std::cout << "maxeval " << maxeval << std::endl;
            std::cout << "cputhreads " << cputhreads << std::endl;
            std::cout << "max_work_packages " << max_work_packages << std::endl;
            std::cout << "cudablocks " << cudablocks << std::endl;
            std::cout << "cudathreadsperblock " << cudathreadsperblock << std::endl;
            std::cout << "devices ";
            for (const int& i : devices)
                std::cout << i << " ";
            std::cout << std::endl;
            std::cout << "n " << n << std::endl;
            std::cout << "m " << m << std::endl;
            std::cout << "total_work_packages " << total_work_packages << std::endl;
            std::cout << "points_per_package " << points_per_package << std::endl;
            std::cout << "r " << m << "*" << r_size << std::endl;
        }
        
        if ( cputhreads == 1 && devices.size() == 1 && devices.count(-1) == 1)
        {
            // Compute serially on cpu
            if (verbosity > 2) std::cout << "computing serially" << std::endl;
            for( U i=0; i < total_work_packages; i++)
            {
                compute(i, z, d, &r[0], r.size()/m, total_work_packages, points_per_package, n, m, func, dim, integralTransform);
            }
        }
        else
        {
            // Create threadpool
            if (verbosity > 2)
            {
                std::cout << "distributing work" << std::endl;
                if ( devices.count(-1) != 0)
                    std::cout << "creating " << std::to_string(cputhreads) << " cputhreads," << std::to_string(extra_threads) << " non-cpu threads" << std::endl;
                else
                    std::cout << "creating " << std::to_string(extra_threads) << " non-cpu threads" << std::endl;
            }
            
            // Setup work queue
            work_queue = total_work_packages;
            
            // Launch worker threads
            U thread_id = 0;
            std::vector<std::thread> thread_pool;
            thread_pool.reserve(cputhreads+extra_threads);
            for (int device : devices)
            {
                if( device != -1)
                {
#ifdef __CUDACC__
                    thread_pool.push_back( std::thread( &Qmc<T,D,U,G>::compute_worker<F1,F2>, this, thread_id, std::cref(z), std::cref(d), std::ref(r), total_work_packages, points_per_package, n, m, std::ref(func), dim, std::ref(integralTransform), device ) ); // Launch non-cpu workers
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
                    thread_pool.push_back( std::thread( &Qmc<T,D,U,G>::compute_worker<F1,F2>, this, thread_id, std::cref(z), std::cref(d), std::ref(r), total_work_packages, points_per_package, n, m, std::ref(func), dim, std::ref(integralTransform), -1 ) ); // Launch cpu workers
                    thread_id += 1;
                }
            }
            // Destroy threadpool
            for( std::thread& thread : thread_pool )
                thread.join();
            thread_pool.clear();
        }
        return reduce(r, n, m);
    };
    
    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::update(result<T,U>& res, U& n, U& m)
    {
        if (verbosity > 2) std::cout << "-- qmc::update called --" << std::endl;
        D errorRatio = computeErrorRatio(res, epsrel, epsabs);
        errorRatio = std::min(errorRatio,20.); // TODO - magic number
        errorRatio = std::max(errorRatio,1.1); // TODO - magic number
        D expectedScaling=0.8; // assume error scales as n^(-expectedScaling) // TODO - magic number
				D additionalMIncreaseFactor = 1.0;  // TODO - magic number
        U newM = minm;
        U newN = getNextN(static_cast<U>(static_cast<D>(n)*pow(errorRatio,1./expectedScaling)));
        if ( newN <= n or ( additionalMIncreaseFactor *errorRatio*errorRatio < static_cast<D>(newN)/static_cast<D>(n)))
        {
            // n did not increase, or increasing m will be faster
            // increase m
            newN = n;
            newM = additionalMIncreaseFactor * static_cast<U>(static_cast<D>(m)*errorRatio*errorRatio);
        }
        if ( maxeval < newN*newM)
        {
            // Decrease n
            newN = getNextN(maxeval/newM);
        }
        n = newN;
        m = newM;
        if(verbosity > 1 ) std::cout << "updated n m " << n << " " << m << std::endl;
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    result<T,U> Qmc<T,D,U,G>::integrate(F1& func, const U dim, F2& integralTransform)
    {
        if ( dim < 1 ) throw std::invalid_argument("qmc::integrate called with dim < 1. Check that your integrand depends on at least one variable of integration.");
        if ( minm < 2 ) throw std::domain_error("qmc::integrate called with minm < 2. This algorithm can not be used with less than 2 random shifts. Please increase minm.");
        if ( max_work_packages == 0 ) throw std::domain_error("qmc::integrate called with max_work_packages = 0. Please set max_work_packages to a positive integer.");

        if (verbosity > 2) std::cout << "-- qmc::integrate called --" << std::endl;
        U n = getNextN(minn); // Increase minn to next available n
        U m = minm;
        if ( maxeval < minn*minm)
        {
            if (verbosity > 2) std::cout << "increasing maxeval " << maxeval << " -> " << minn*minm << std::endl;
            maxeval = minn*minm;
        }
        result<T,U> res;
        do
        {
            if (verbosity > 1) std::cout << "iterating" << std::endl;
            res = sample(func,dim,integralTransform,n,m);
            if (verbosity > 1) std::cout << "result " << res.integral << " " << res.error << std::endl;
            update(res,n,m);
        } while  ( computeErrorRatio(res,epsrel,epsabs) > 1. && (res.n*res.m) < maxeval ); // TODO - if error estimate is not decreasing quit
        return res;
    };
    
    template <typename T, typename D, typename U, typename G>
    template <typename F1>
    result<T,U> Qmc<T,D,U,G>::integrate(F1& func, const U dim)
    {
        integrators::Korobov3<D,U> defaultIntegralTransform;
        return integrate(func, dim, defaultIntegralTransform);
    };
    
    template <typename T, typename D, typename U, typename G>
    Qmc<T,D,U,G>::Qmc() :
    randomGenerator( G( std::random_device{}() ) ), minn(8191), minm(32), epsrel(std::numeric_limits<D>::max()), epsabs(std::numeric_limits<D>::max()), maxeval(std::numeric_limits<U>::max()), max_work_packages(2560000), cputhreads(std::thread::hardware_concurrency()), cudablocks(1000), cudathreadsperblock(256), devices({-1}), verbosity(0)
    {
        // Check U satisfies requirements of mod_mul implementation
        static_assert( std::numeric_limits<U>::is_modulo, "Qmc integrator constructed with a type U that is not modulo. Please use a different unsigned integer type for U.");
        static_assert( std::numeric_limits<D>::radix == 2, "Qmc integrator constructed with a type D that does not have radix == 2. Please use a different floating point type for D.");
        
        if ( cputhreads == 0 ) cputhreads = 1; // Correct cputhreads if hardware_concurrency is 0, i.e. not well defined or not computable
        
        // TODO - get number of cuda devices and populate
        
        initg();
    };
    
};
