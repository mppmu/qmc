#ifndef QMC_TRANSFORMS_DETAIL_KOROBOV_TERM_H
#define QMC_TRANSFORMS_DETAIL_KOROBOV_TERM_H

#include <type_traits> // enable_if

#include "binomial.hpp"
#include "korobov_coefficient.hpp"

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Korobov Transform Terms
             */
            template<typename D, typename U, U k, U a, U b, typename = void>
            struct KorobovTerm
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                constexpr static D value(const D& x)
                {
                    return KorobovTerm<D,U,k-1,a,b>::value(x)*x+KorobovCoefficient<D,U,b-k,a,b>::value;
                }
            };
            template<typename D, typename U, U k, U a, U b>
            struct KorobovTerm<D, U, k, a, b, typename std::enable_if<k == 0>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                constexpr static D value(const D& x)
                {
                    return KorobovCoefficient<D,U,b,a,b>::value;
                }
            };
        };
    };
};

#endif
