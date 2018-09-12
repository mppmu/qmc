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
            template<typename U, U n, typename = void>
            struct Factorial
            {
                constexpr static U value = n*Factorial<U,n-1>::value;
            };

            template<typename U, U n>
            struct Factorial<U, n, typename std::enable_if<n == 0>::type>
            {
                constexpr static U value = U(1);
            };

            template<typename U, U n, typename T>
            constexpr U Factorial<U,n,T>::value;
        };
    };
};

#endif
