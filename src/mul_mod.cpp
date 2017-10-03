#include "mul_mod.hpp"

#include <type_traits> // make_signed

namespace qmcutil {

    template <typename R, typename D, typename U>
    R mul_mod(U a, U b, U k)
    {
        // Computes: (a*b % k) correctly even when a*b overflows std::numeric_limits<typename std::make_signed<U>>
        // Assumes:
        // 1) std::numeric_limits<U>::is_modulo
        // 2) a < k
        // 3) b < k
        // 4) k < std::numeric_limits<typename std::make_signed<U>::type>::max()
        // 5) k < pow(std::numeric_limits<D>::radix,std::numeric_limits<D>::digits-1)
        using S = typename std::make_signed<U>::type;
        D x = static_cast<D>(a);
        U c = static_cast<U>( (x*b) / k );
        S r = static_cast<S>( (a*b) - (c*k) ) % static_cast<S>(k);
        return static_cast<R>(r < 0 ? (r+k) : r);
    };

};
