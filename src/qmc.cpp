#include "qmc.hpp"

#include <cmath> // modf, abs, sqrt
#include <stdexcept> // domain_error, invalid_argument
#include <thread> // thread
#include <algorithm> // min

#include "qmc_default.cpp"
#include "qmc_complex.cpp"

namespace integrators {

    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::integralTransform(std::vector<D>& x, D& wgt, const U dim) const
    {
        // Korobov r = 3
        for (U sDim = 0; sDim < dim; sDim++)
        {
            wgt*=x[sDim]*x[sDim]*x[sDim]*140.*(1.-x[sDim])*(1.-x[sDim])*(1.-x[sDim]);
            x[sDim]=x[sDim]*x[sDim]*x[sDim]*x[sDim]*(35.+x[sDim]*(-84.+x[sDim]*(70.+x[sDim]*(-20.))));
            // Note: loss of precision can cause x > 1., must keep in x \elem [0,1]
            if ( x[sDim] > 1. )
                x[sDim] = 1.;
        }
    };

    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initzn(std::vector<U>& z, U& n, const U dim) const
    {
        n = getN();
        z = generatingVectors.lower_bound(n)->second;

        if ( dim > z.size() ) throw std::domain_error("dim > generating vector dimension. Please supply a generating vector table with a larger number of dimensions");

        z.resize(dim);
    };

    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initd(std::vector<D>& d, const U dim)
    {
        d.clear();
        for (U k = 0; k < m; k++)
            for (U sDim = 0; sDim < dim; sDim++)
                d.push_back(uniformDistribution(randomGenerator));
    };

    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initr(std::vector<T>& r, const U block) const
    {
        r.clear();
        r.resize(block * m, {0.});
    };

    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::initzdrn(std::vector<U>& z, std::vector<D>& d, std::vector<T>& r, U& n, U& blocks, U& block, const U dim)
    {
        initzn(z, n, dim);
        initd(d, dim);

        // Set size (block) and number of blocks (blocks) of points to be computed simultaneously
        block = blockSize;
        block = std::min(block, n);
        blocks = n/block;
        if ( n % block != 0 ) blocks++;

        initr(r, block);
    };

    template <typename T, typename D, typename U, typename G>
    result<T> Qmc<T,D,U,G>::reduce(const std::vector<T>& r, const U n, const U block) const
    {
        T mean = {0.};
        T variance = {0.};
        for(U k = 0; k < m; k++)
        {
            T sum = {0.};
            T delta = {0.};
            T kahanC = {0.};
            for (U i = 0; i<block; i++)
            {
                // Compute sum using Kahan summation
                // equivalent to: sum += r.at(k*block+i);
                T kahanY = r.at(k*block+i) - kahanC;
                T kahanT = sum + kahanY;
                T kahanD = kahanT - sum;
                kahanC = kahanD - kahanY;
                sum = kahanT;
            }

            // Compute Variance using online algorithm (Knuth, The Art of Computer Programming)
            delta = sum - mean;
            mean = mean + delta/(static_cast<T>(k+1));

            variance = computeVariance(mean, variance, sum, delta);

        }
        T integral = mean/(static_cast<T>(n));
        variance = variance/( static_cast<T>(m-1) * static_cast<T>(m) * static_cast<T>(n) * static_cast<T>(n) ); // variance of the mean
        T error = computeError(variance);

        return {integral, error};
    };

    template <typename T, typename D, typename U, typename G>
    void Qmc<T,D,U,G>::compute(const U blocks, const U block, const U k, const U i, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U n, const std::function<T(D[])>& func, const U dim) const
    {
        T kahanC = {0.};
        for( U b = 0; b < blocks; b++ )
        {
            U offset = b * block;
            if(offset+i < n)
            {
                D wgt = 1.;
                D mynull = 0;
                std::vector<D> x(dim, 0.);

                // TODO, prevent needless overflow on this line by using modular arithmetic
                for (U sDim = 0; sDim < dim; sDim++)
                    x[sDim] = std::modf(static_cast<D>(i+offset)*static_cast<D>(z.at(sDim))/(static_cast<D>(n)) + d.at(k*dim+sDim), &mynull);

                integralTransform(x, wgt, dim);

                T point = func(x.data());
                if ( computeIsFinite(point, wgt) )
                {
                    // Compute sum using Kahan summation
                    // equivalent to: r.at(k*block+i) += wgt*point;
                    T kahanY = wgt*point - kahanC;
                    T kahanT = r.at(k*block+i) + kahanY;
                    T kahanD = kahanT - r.at(k*block+i);
                    kahanC = kahanD - kahanY;
                    r.at(k*block+i) = kahanT;
                }
            }
        }
    };

    template <typename T, typename D, typename U, typename G>
    U Qmc<T,D,U,G>::getN() const
    {
        U n;

        if ( generatingVectors.lower_bound(minN) == generatingVectors.end() )
        {
            n = generatingVectors.rbegin()->first;
            throw std::domain_error("Qmc integrator does not have generating vector with n larger than the requested minN. Please decrease minN to less than " + std::to_string(n) + " or provide a generating vector with a larger n.");
        }

        n = generatingVectors.lower_bound(minN)->first;

        return n;
    };

    template <typename T, typename D, typename U, typename G>
    result<T> Qmc<T,D,U,G>::integrate(const std::function<T(D[])>& func, const U dim)
    {
        if ( dim < 1 ) throw std::invalid_argument("Qmc integrator constructed with dim < 1. Check that your integrand depends on at least one variable of integration.");
        if ( m < 2 ) throw std::domain_error("Qmc integrator called with m < 2. This algorithm can not be used with less than 2 random shifts. Please increase m.");

        // Set block and blocks, generate z, d, r. Increase n if it does not match any generating vector.
        std::vector<U> z;
        std::vector<D> d;
        std::vector<T> r;
        U n;
        U blocks;
        U block;
        initzdrn(z, d, r, n, blocks, block, dim);

        for (U k = 0; k < m; k++)
        {
            std::vector<std::thread> threads;
            threads.reserve(block);
            for( U i = 0; i < block; i++)
            {
                if(block == 1)
                {
                    compute(blocks, block, k, i, std::ref(z), std::ref(d), std::ref(r), n, std::ref(func), dim); // Compute serially

                } else
                {
                    threads.push_back( std::thread( &Qmc<T,D,U,G>::compute, this, blocks, block, k, i, std::ref(z), std::ref(d), std::ref(r), n, std::ref(func), dim ) ); // Compute in parallel
                }
            }
            for( std::thread& thread : threads )
                thread.join();
            threads.clear();
            threads.reserve(block);
        }

        return reduce(r, n, block);
    };

    template <typename T, typename D, typename U, typename G>
    Qmc<T,D,U,G>::Qmc() :
    randomGenerator( G((std::random_device())()) ), minN(8191), m(32), blockSize(std::thread::hardware_concurrency())
    {
        if ( blockSize == 0 ) blockSize = 1; // Correct blockSize if hardware_concurrency is 0, i.e. not well defined or not computable
        initg();
    };
    
};
