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
            const U number_of_integration_variables;

            NoneImpl(I f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            }
            void evaluate(D* x, decltype(f(x))* res, U count)
            {
                for (U i = 0; i!= count; ++i) {
                    res[i] = f(x + i);
                }
            }
        };
        struct None
        {
            template<typename I, typename D, U M> using type = NoneImpl<I, D>;
        };
    };
};

#endif
