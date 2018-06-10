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
            template<typename D, typename U, U k, U a, U b, typename = void>
            struct KorobovCoefficient
            {
                constexpr static D value = (D(-1)*(D(b)-D(k)+D(1))*D(KorobovCoefficient<D,U,k-1,a,b>::value)*(D(a)+D(k)))/(D(k)*(D(a)+D(k)+D(1)));
            };

            template<typename D, typename U, U k, U a, U b>
            struct KorobovCoefficient<D, U, k, a, b, typename std::enable_if<k == 0>::type>
            {
                constexpr static D value = ((D(2)*D(b)+D(1))*D(Binomial<U,2*b,b>::value))/(D(a)+D(1));
            };
        };
    };
};

#endif
