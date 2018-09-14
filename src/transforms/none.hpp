#ifndef QMC_TRANSFORMS_NONE_H
#define QMC_TRANSFORMS_NONE_H

#include <cstddef> // nullptr_t

namespace integrators
{
    namespace transforms
    {
        template<typename I, typename D>
        struct NoneImpl
        {
            I f; // original function
            const U dim;

            NoneImpl(I f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            }
        };
        struct None
        {
            template<typename I, typename D> using type = NoneImpl<I, D>;
        };
    };
};

#endif
