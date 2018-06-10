#ifndef QMC_TRANSFORMS_DETAIL_BINOMIAL_H
#define QMC_TRANSFORMS_DETAIL_BINOMIAL_H

#include <type_traits> // enable_if

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Binomial Coefficients: Binomial<U,n,k>::value gives the type U binomial coefficient (n k)
             */
            template<typename U, U n, U k, typename = void>
            struct Binomial
            {
                constexpr static U value = (Binomial<U,n-1,k-1>::value + Binomial<U,n-1,k>::value);
            };

            // TODO - optimisation
            // k > n -k ? bin(n,n-k) : bin(n,k)

            template<typename U, U n, U k>
            struct Binomial<U, n, k, typename std::enable_if<n < k>::type>
            {
                constexpr static U value = 0;
            };

            template<typename U, U n, U k>
            struct Binomial<U, n, k, typename std::enable_if<k == 0>::type>
            {
                constexpr static U value = 1;
            };

            template<typename U, U n, U k>
            struct Binomial<U, n, k, typename std::enable_if<n == k && k != 0>::type>
            {
                constexpr static U value = 1;
            };
        };
    };
};

#endif
