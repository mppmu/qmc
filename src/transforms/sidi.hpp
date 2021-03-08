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
         * Sidi Transform: Sidi<D,r>(func) takes the weight r Sidi transform of func
         */
        template<typename I, typename D, U r, typename = void>
        struct SidiImpl{};

        // Odd r
        template<typename I, typename D, U r>
        struct SidiImpl<I, D, r, typename std::enable_if<(r % 2) != 0 && (r != 0)>::type>
        {
            I f; // original function
            const U number_of_integration_variables;
            const D pi = acos( D(-1) );

            SidiImpl(I f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                using std::sin;

                D wgt = 1;

                const D fac1 = detail::Factorial<r>::value;
                const D fac2 = detail::Factorial<(r-U(1))/U(2)>::value;

                const D wgt_prefactor = pi/detail::IPow<D,r>::value(D(2))*fac1/fac2/fac2;
                const D transform_prefactor = D(1)/detail::IPow<D,U(2)*r-U(1)>::value(D(2))*fac1/fac2/fac2;
                for(U s = 0; s<number_of_integration_variables; s++)
                {
                    wgt *= wgt_prefactor*detail::IPow<D,r>::value(sin(pi*x[s]));
                    x[s] = transform_prefactor*detail::SidiTerm<D,(r-U(1))/U(2),r>::value(x[s],pi);
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
                return wgt * f(x);
            }
            void evaluate(D* x, decltype(f(x))* res, U count)
            {
                auto xx = x;
                D* wgts = new D[count];
                for (U i = 0; i!= count; ++i, xx+=number_of_integration_variables) {
                    wgts[i] = 1;
                    const D fac1 = detail::Factorial<r>::value;
                    const D fac2 = detail::Factorial<(r-U(1))/U(2)>::value;

                    const D wgt_prefactor = pi/detail::IPow<D,r>::value(D(2))*fac1/fac2/fac2;
                    const D transform_prefactor = D(1)/detail::IPow<D,U(2)*r-U(1)>::value(D(2))*fac1/fac2/fac2;
                    for(U s = 0; s<number_of_integration_variables; s++)
                    {
                        wgts[i] *= wgt_prefactor*detail::IPow<D,r>::value(sin(pi*xx[s]));
                        xx[s] = transform_prefactor*detail::SidiTerm<D,(r-U(1))/U(2),r>::value(xx[s],pi);
                        // loss of precision can cause xx < 0 or xx > 1 must keep in xx \elem [0,1]
                        if (xx[s] > D(1)) xx[s] = D(1);
                        if (xx[s] < D(0)) xx[s] = D(0);
                    }
                }
                f.evaluate(x, res, count);
                for (U i = 0; i!= count; ++i, xx+=number_of_integration_variables) {
                    res[i] = wgts[i] * res[i];
                }
                delete[] wgts;
            }
        };

        // Even r
        template<typename I, typename D, U r>
        struct SidiImpl<I, D, r, typename std::enable_if<(r % 2) == 0 && (r != 0)>::type>
        {
            I f; // original function
            const U number_of_integration_variables;
            const D pi = acos( D(-1) );

            SidiImpl(I f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                using std::sin;

                D wgt = 1;

                const D fac1 = detail::Factorial<r/U(2)-U(1)>::value;
                const D fac2 = detail::Factorial<r-U(1)>::value;

                const D wgt_prefactor = detail::IPow<D,r-U(2)>::value(D(2))*D(r)*fac1*fac1/fac2;
                const D transform_prefactor = D(r)/D(2)/pi*fac1*fac1/fac2;
                for(U s = 0; s<number_of_integration_variables; s++)
                {
                    wgt *= wgt_prefactor*detail::IPow<D,r>::value(sin(pi*x[s]));
                    x[s] = transform_prefactor*detail::SidiTerm<D,r/U(2)-U(1),r>::value(x[s],pi);
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
                return wgt * f(x);
            }
            void evaluate(D* x, decltype(f(x))* res, U count)
            {
                auto xx = x;
                D* wgts = new D[count];
                for (U i = 0; i!= count; ++i, xx+=number_of_integration_variables) {
                    wgts[i] = 1;
                    const D fac1 = detail::Factorial<r/U(2)-U(1)>::value;
                    const D fac2 = detail::Factorial<r-U(1)>::value;

                    const D wgt_prefactor = detail::IPow<D,r-U(2)>::value(D(2))*D(r)*fac1*fac1/fac2;
                    const D transform_prefactor = D(r)/D(2)/pi*fac1*fac1/fac2;
                    for(U s = 0; s<number_of_integration_variables; s++)
                    {
                        wgts *= wgt_prefactor*detail::IPow<D,r>::value(sin(pi*xx[s]));
                        xx[s] = transform_prefactor*detail::SidiTerm<D,r/U(2)-U(1),r>::value(xx[s],pi);
                        // loss of precision can cause xx < 0 or xx > 1 must keep in xx \elem [0,1]
                        if (xx[s] > D(1)) xx[s] = D(1);
                        if (xx[s] < D(0)) xx[s] = D(0);
                    }
                }
                f.evaluate(x, res, count);
                for (U i = 0; i!= count; ++i, xx+=number_of_integration_variables) {
                    res[i] = wgts[i] * res[i];
                }
                delete[] wgts;
            }
        };

        // r == 0
        template<typename I, typename D, U r>
        struct SidiImpl<I, D, r, typename std::enable_if<r == 0>::type>
        {
            I f; // original function
            const U number_of_integration_variables;

            SidiImpl(I f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            }
            void evaluate(D* x, decltype(f(x))* res, U count)
            {
                f.evaluate(x, res, count);
            }
        };

        template<U r0>
        struct Sidi
        {
            template<typename I, typename D, U M> using type = SidiImpl<I, D, r0>;
        };

    };
};

#endif
