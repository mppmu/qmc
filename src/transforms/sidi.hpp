#ifndef QMC_TRANSFORMS_SIDI_H
#define QMC_TRANSFORMS_SIDI_H

#include <type_traits> // enable_if
#include <cmath> // sin, acos

#include "detail/binomial.hpp"
#include "detail/factorial.hpp"
#include "detail/ipow.hpp"
#include "detail/sidi_coefficient.hpp"
#include "detail/sidi_term.hpp"

namespace integrators
{
    namespace transforms
    {
        /*
         * Sidi Transform: Sidi<D,U,r>(func) takes the weight r Sidi transform of func
         */
        template<typename F1, typename D, typename U, U r, typename = void>
        struct Sidi{};

        // Odd r
        template<typename F1, typename D, typename U, U r>
        struct Sidi<F1, D, U, r, typename std::enable_if<(r % 2) != 0 && (r != 0)>::type>
        {
            F1 f; // original function
            const U dim;
            const D pi = acos( D(-1) );

            Sidi(F1 f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                using std::sin;

                D wgt = 1;

                const D fac1 = detail::Factorial<U,r>::value;
                const D fac2 = detail::Factorial<U,(r-U(1))/U(2)>::value;

                const D wgt_prefactor = pi/detail::IPow<D,U,r>::value(D(2))*fac1/fac2/fac2;
                const D transform_prefactor = D(1)/detail::IPow<D,U,U(2)*r-U(1)>::value(D(2))*fac1/fac2/fac2;
                for(U s = 0; s<dim; s++)
                {
                    wgt *= wgt_prefactor*detail::IPow<D,U,r>::value(sin(pi*x[s]));
                    x[s] = transform_prefactor*detail::SidiTerm<D,U,(r-U(1))/U(2),r>::value(x[s],pi);
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
                return wgt * f(x);
            }
        };

        // Even r
        template<typename F1, typename D, typename U, U r>
        struct Sidi<F1, D, U, r, typename std::enable_if<(r % 2) == 0 && (r != 0)>::type>
        {
            F1 f; // original function
            const U dim;
            const D pi = acos( D(-1) );

            Sidi(F1 f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                using std::sin;

                D wgt = 1;

                const D fac1 = detail::Factorial<U,r/U(2)-U(1)>::value;
                const D fac2 = detail::Factorial<U,r-U(1)>::value;

                const D wgt_prefactor = detail::IPow<D,U,r-U(2)>::value(D(2))*D(r)*fac1*fac1/fac2;
                const D transform_prefactor = D(r)/D(2)/pi*fac1*fac1/fac2;
                for(U s = 0; s<dim; s++)
                {
                    wgt *= wgt_prefactor*detail::IPow<D,U,r>::value(sin(pi*x[s]));
                    x[s] = transform_prefactor*detail::SidiTerm<D,U,r/U(2)-U(1),r>::value(x[s],pi);
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
                return wgt * f(x);
            }
        };

        // r == 0
        template<typename F1, typename D, typename U, U r>
        struct Sidi<F1, D, U, r, typename std::enable_if<r == 0>::type>
        {
            F1 f; // original function
            const U dim;

            Sidi(F1 f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            }
        };

    };
};

#endif
