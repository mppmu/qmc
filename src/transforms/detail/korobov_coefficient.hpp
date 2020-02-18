#ifndef QMC_TRANSFORMS_DETAIL_KOROBOV_COEFFICIENT_H
#define QMC_TRANSFORMS_DETAIL_KOROBOV_COEFFICIENT_H

#include <type_traits> // enable_if

#include "binomial.hpp"

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Korobov Coefficients
             */
            template<typename D, U k, U a, U b, typename = void>
            struct KorobovCoefficient
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                const static D value()
                {
                    return (D(-1)*(D(b)-D(k)+D(1))*D(KorobovCoefficient<D,k-1,a,b>::value())*(D(a)+D(k)))/(D(k)*(D(a)+D(k)+D(1)));
                }
            };

            template<typename D, U k, U a, U b>
            struct KorobovCoefficient<D, k, a, b, typename std::enable_if<k == 0>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                const static D value()
                {
                    return ((D(a)+D(b)+D(1))*D(Binomial<a+b,b>::value))/(D(a)+D(1));
                }
            };
        };
    };
};

#endif
