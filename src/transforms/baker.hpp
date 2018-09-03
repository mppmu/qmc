#ifndef QMC_TRANSFORMS_BAKER_H
#define QMC_TRANSFORMS_BAKER_H

namespace integrators
{
    namespace transforms
    {
        template<typename F1, typename D, typename U = unsigned long long int>
        struct Baker
        {
            F1 f; // original function
            const U dim;

            Baker(F1 f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                D wgt = 1;
                for (U s = 0; s < dim; s++)
                {
                    x[s] = D(1) - fabs(D(2)*x[s]-D(1)) ;
                    // loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1]
                    if (x[s] > D(1)) x[s] = D(1);
                    if (x[s] < D(0)) x[s] = D(0);
                }
                return wgt * f(x);
            }
        };
    };
};

#endif
