#ifndef QMC_TRANSFORMS_DETAIL_SIDI_COEFFICIENT_H
#define QMC_TRANSFORMS_DETAIL_SIDI_COEFFICIENT_H

#include <type_traits> // enable_if

#include "binomial.hpp"
#include "ipow.hpp"

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Sidi Coefficients
             */
            template<typename D, U k, U r, typename = void>
            struct SidiCoefficient{};

            // Odd r
            template<typename D, U k, U r>
            struct SidiCoefficient<D, k, r, typename std::enable_if<(r % 2) != 0>::type>
            {
                const static D value()
                {
                    return IPow<D,(r-U(1))/U(2)-k>::value(D(-1))*D(Binomial<r,k>::value)/(D(2)*k-r);
                }
            };

            // Even r
            template<typename D, U k, U r>
            struct SidiCoefficient<D, k, r, typename std::enable_if<(r % 2) == 0>::type>
            {
                const static D value()
                {
                    return IPow<D,r/U(2)-k>::value(D(-1))*D(Binomial<r,k>::value)/(D(2)*k-r);
                }
            };
        };
    };
};

#endif
