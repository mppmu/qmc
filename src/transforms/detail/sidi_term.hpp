#ifndef QMC_TRANSFORMS_DETAIL_SIDI_TERM_H
#define QMC_TRANSFORMS_DETAIL_SIDI_TERM_H

#include <type_traits> // enable_if
#include <cmath> // cos, sin

#include "binomial.hpp"
#include "sidi_coefficient.hpp"

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Sidi Transform Terms
             */
            template<typename D, U k, U r, typename = void>
            struct SidiTerm{};

            // Odd r
            template<typename D, U k, U r>
            struct SidiTerm<D, k, r, typename std::enable_if<(r % 2) != 0 && (k != 0)>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                const static D value(const D& x, const D pi)
                {
                    using std::cos;
                    return SidiCoefficient<D,k,r>::value()*(cos((D(2)*D(k)-D(r))*pi*x) - D(1)) + SidiTerm<D,k-U(1),r>::value(x,pi);
                }
            };
            template<typename D, U k, U r>
            struct SidiTerm<D, k, r, typename std::enable_if<((r % 2) != 0) && (k == 0)>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                const static D value(const D& x, const D pi)
                {
                    using std::cos;
                    return SidiCoefficient<D,0,r>::value()*(cos(-D(r)*pi*x) - D(1));
                }
            };

            // Even r
            template<typename D, U k, U r>
            struct SidiTerm<D, k, r, typename std::enable_if<(r % 2) == 0 && (k != 0)>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                const static D value(const D& x, const D pi)
                {
                    using std::sin;
                    return SidiCoefficient<D,k,r>::value()*sin((D(2)*D(k)-D(r))*pi*x) + SidiTerm<D,k-U(1),r>::value(x,pi);
                }
            };
            template<typename D, U k, U r>
            struct SidiTerm<D, k, r, typename std::enable_if<((r % 2) == 0) && (k == 0)>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                const static D value(const D& x, const D pi)
                {
                    using std::sin;
                    return SidiCoefficient<D,0,r>::value()*sin(-D(r)*pi*x) + D(1)/D(2)*pi*Binomial<r,r/2>::value*x;
                }
            };

        };
    };
};

#endif
