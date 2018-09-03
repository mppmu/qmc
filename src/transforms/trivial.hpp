#ifndef QMC_TRANSFORMS_TRIVIAL_H
#define QMC_TRANSFORMS_TRIVIAL_H

namespace integrators
{
    namespace transforms
    {
        template<typename F1, typename D, typename U = unsigned long long int>
        struct Trivial
        {
            F1 f; // original function
            U dim;

            Trivial(F1 f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            };
        };

    };

};

#endif
