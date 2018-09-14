#ifndef QMC_MATH_MUL_MOD_H
#define QMC_MATH_MUL_MOD_H

#include <type_traits> // make_signed

namespace integrators
{
    namespace math
    {
        template <typename R, typename D>
#ifdef __CUDACC__
        __host__ __device__
#endif
        R mul_mod(U a, U b, U k)
        {
            // Computes: (a*b % k) correctly even when a*b overflows std::numeric_limits<typename std::make_signed<U>>
            // Assumes:
            // 1) std::numeric_limits<U>::is_modulo
            // 2) a < k
            // 3) b < k
            // 4) k < std::numeric_limits<typename std::make_signed<U>::type>::max()
            // 5) k < std::pow(std::numeric_limits<D>::radix,std::numeric_limits<D>::digits-1)
            using S = typename std::make_signed<U>::type;
            D x = static_cast<D>(a);
            U c = static_cast<U>( (x*b) / k );
            S r = static_cast<S>( (a*b) - (c*k) ) % static_cast<S>(k);
            return r < 0 ? static_cast<R>(static_cast<U>(r)+k) : static_cast<R>(r);
        };
    };
};

#endif
