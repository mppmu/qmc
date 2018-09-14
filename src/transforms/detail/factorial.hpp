#ifndef QMC_TRANSFORMS_DETAIL_FACTORIAL_H
#define QMC_TRANSFORMS_DETAIL_FACTORIAL_H

#include <type_traits> // enable_if

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Factorial: Factorial<U,n>::value gives the type U factorial of n
             */
            template<U n, typename = void>
            struct Factorial
            {
                constexpr static U value = n*Factorial<n-1>::value;
            };

            template<U n>
            struct Factorial<n, typename std::enable_if<n == 0>::type>
            {
                constexpr static U value = U(1);
            };

            template<U n, typename T>
            constexpr U Factorial<n,T>::value;
        };
    };
};

#endif
