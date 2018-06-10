#ifndef QMC_TRANSFORMS_TRIVIAL_H
#define QMC_TRANSFORMS_TRIVIAL_H

namespace integrators
{
    namespace transforms
    {
        template<typename D, typename U = unsigned long long int>
        struct Trivial
        {
#ifdef __CUDACC__
            __host__ __device__
#endif
            void operator()(D* x, D& wgt, const U dim) const {}
        };

    };

};

#endif
