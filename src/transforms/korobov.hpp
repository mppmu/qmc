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
         * Korobov Transform: Korobov<D,U,r>(x,weight,dim) takes the weight r Korobov transform of x
         */
        template<typename D, typename U, U r>
        struct Korobov
        {
#ifdef __CUDACC__
            __host__ __device__
#endif
            void operator()(D* x, D& wgt, const U dim) const
            {
                const D prefactor = (D(2)*r+D(1))*detail::Binomial<U,2*r,r>::value;
                for(U s = 0; s<dim; s++)
                {
                    wgt *= prefactor*detail::ipow<D,U,r>::value(x[s])*detail::ipow<D,U,r>::value(D(1)-x[s]);
                    x[s] = detail::ipow<D,U,r+1>::value(x[s])*detail::KorobovTerm<D,U,r,r,r>::value(x[s]);
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
            }
        };
    };
};

#endif
