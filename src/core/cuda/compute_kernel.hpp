#ifndef QMC_CORE_CUDA_COMPUTE_KERNEL_H
#define QMC_CORE_CUDA_COMPUTE_KERNEL_H

#ifdef __CUDACC__

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            // TODO - make use of restricted pointers?
            template <typename T, typename D, typename U, typename F1, typename F2>
            __global__
            void compute_kernel(const U work_offset, const U work_this_iteration, const U total_work_packages, const U* z, const D* d, T* r, const U d_r_size_over_m, const U n, const U m, F1* func, const U dim, F2* integral_transform)
            {
                U i = blockIdx.x*blockDim.x + threadIdx.x;
                if (i < work_this_iteration)
                {
                    for (U k = 0; k < m; k++)
                    {
                        T kahan_c = {0.};
                        for (U offset = work_offset + i; offset < n; offset += total_work_packages)
                        {
                            D wgt = 1.;
                            D mynull = 0;
                            D x[25]; // TODO - template parameter?

                            for (U sDim = 0; sDim < dim; sDim++)
                            {
                                x[sDim] = modf(integrators::math::mul_mod<D, D, U>(offset, z[sDim], n) / n + d[k*dim + sDim], &mynull);
                            }

                            (*integral_transform)(x, wgt, dim);

                            T point = (*func)(x);

                            // Compute sum using Kahan summation
                            // equivalent to: r[k*d_r_size_over_m + i] += wgt*point;
                            T kahan_y = wgt*point - kahan_c;
                            T kahan_t = r[k*d_r_size_over_m + i] + kahan_y;
                            T kahan_d = kahan_t - r[k*d_r_size_over_m + i];
                            kahan_c = kahan_d - kahan_y;
                            r[k*d_r_size_over_m + i] = kahan_t;
                        }
                    }
                }
            };
        };
    };
};

#endif
#endif
