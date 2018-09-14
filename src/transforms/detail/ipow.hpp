#ifndef QMC_TRANSFORMS_DETAIL_IPOW_H
#define QMC_TRANSFORMS_DETAIL_IPOW_H

#include <type_traits> // enable_if

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Power function: IPow<D,i>(d) raises the D d to the U power i
             */
            template<typename D, U n, typename = void>
            struct IPow // n%2 == 0 && n != 0
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                static D value(D base)
                {
                    D power = IPow<D,n/2>::value(base);
                    return power * power;
                }
            };
            template<typename D, U n>
            struct IPow<D, n, typename std::enable_if< n%2 != 0 && n != 0>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                static D value(D base)
                {
                    D power = IPow<D,(n-1)/2>::value(base);
                    return base * power * power;
                }
            };
            template<typename D, U n>
            struct IPow<D, n, typename std::enable_if< n == 0>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                static D value(D base)
                {
                    return D(1);
                }
            };
        };
    };
};

#endif
