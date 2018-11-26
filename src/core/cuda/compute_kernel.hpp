#ifndef QMC_CORE_CUDA_COMPUTE_KERNEL_H
#define QMC_CORE_CUDA_COMPUTE_KERNEL_H

#ifdef __CUDACC__

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            // TODO (V2) - investigate if using restricted pointers improves performance
            template <U M, typename T, typename D, typename I>
            __global__
            void compute_kernel(const U work_offset, const U work_this_iteration, const U total_work_packages, const U* z, const D* d, T* r, const U d_r_size_over_m, const U n, const U m, I* func)
            {
                U i = blockIdx.x*blockDim.x + threadIdx.x;
                if (i < work_this_iteration)
                {
                    for (U k = 0; k < m; k++)
                    {
                        for (U offset = work_offset + i; offset < n; offset += total_work_packages)
                        {
                            D wgt = 1.;
                            D mynull = 0;
                            D x[M];

                            for (U sDim = 0; sDim < func->number_of_integration_variables; sDim++)
                            {
                                x[sDim] = modf(integrators::math::mul_mod<D, D>(offset, z[sDim], n) / n + d[k*func->number_of_integration_variables + sDim], &mynull);
                            }

                            T point = (*func)(x);
                            
                            r[k*d_r_size_over_m + i] += wgt*point;
                        }
                    }
                }
            };

            template <U M, typename T, typename D, typename I>
            __global__
            void generate_samples_kernel(const U work_offset, const U work_this_iteration, const U* z, const D* d, T* r, const U n, I* func)
            {
                U i = blockIdx.x*blockDim.x + threadIdx.x;
                if (i < work_this_iteration)
                {
                    D mynull = 0;
                    D x[M];

                    for (U sDim = 0; sDim < func->number_of_integration_variables; sDim++)
                    {
                        x[sDim] = modf(integrators::math::mul_mod<D, D>(work_offset + i, z[sDim], n) / n + d[sDim], &mynull);
                    }

                    r[i] = (*func)(x);
                }
            };
        };
    };
};

#endif
#endif
