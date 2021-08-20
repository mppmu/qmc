#ifndef QMC_TRANSFORMS_KOROBOV_H
#define QMC_TRANSFORMS_KOROBOV_H

#include <type_traits> // integral_constant

#include "detail/ipow.hpp"
#include "detail/binomial.hpp"
#include "detail/korobov_coefficient.hpp"
#include "detail/korobov_term.hpp"

namespace integrators
{
    namespace transforms
    {
        /*
         * Korobov Transform: Korobov<D,r0,r1>(func) takes the weight r0,r1 Korobov transform of func
         */
        template<typename I, typename D, U r0, U r1 = r0>
        struct KorobovImpl
        {
            I f; // original function
            const U number_of_integration_variables;

            KorobovImpl(I f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                D wgt = 1;
                const D prefactor = (D(r0)+D(r1)+D(1))*detail::Binomial<r0+r1,r0>::value;
                for(U s = 0; s<number_of_integration_variables; s++)
                {
                    wgt *= prefactor*detail::IPow<D,r0>::value(x[s])*detail::IPow<D,r1>::value(D(1)-x[s]);
                    x[s] = detail::IPow<D,r0+1>::value(x[s])*detail::KorobovTerm<D,r1,r0,r1>::value(x[s]);
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
                return wgt * f(x);
            }
            void operator()(D* x, decltype(f(x))* res, U count)
            {
                if constexpr (hasBatching<I, D*, decltype(f(x))*, U>(0)) {
                    auto xx = x;
                    D* wgts = new D[count];
                    for (U i = 0; i!= count; ++i, xx+=number_of_integration_variables) {
                        wgts[i] = 1;
                        const D prefactor = (D(r0)+D(r1)+D(1))*detail::Binomial<r0+r1,r0>::value;
                        for(U s = 0; s<number_of_integration_variables; s++)
                        {
                            wgts[i] *= prefactor*detail::IPow<D,r0>::value(xx[s])*detail::IPow<D,r1>::value(D(1)-xx[s]);
                            xx[s] = detail::IPow<D,r0+1>::value(xx[s])*detail::KorobovTerm<D,r1,r0,r1>::value(xx[s]);
                            // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                            if (xx[s] > D(1)) xx[s] = D(1);
                            if (xx[s] < D(0)) xx[s] = D(0);
                        }
                    }
                    f(x, res, count);
                    for (U i = 0; i!= count; ++i, xx+=number_of_integration_variables) {
                        res[i] = wgts[i] * res[i];
                    }
                    delete[] wgts;
                } else {
                    for (U i = U(); i != count; ++i) {
                        res[i] = operator()(x + i * f.number_of_integration_variables);
                    }
                }
            }

        };
        template<U r0, U r1 = r0>
        struct Korobov
        {
            template<typename I, typename D, U M> using type = KorobovImpl<I, D, r0, r1>;
        };
    };
};

#endif
