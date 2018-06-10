#ifndef QMC_CORE_CUDA_COMPUTE_H
#define QMC_CORE_CUDA_COMPUTE_H

#ifdef __CUDACC__
#include <memory> // unique_ptr
#include <cuda_runtime_api.h> // cudadDeviceSynchronize

#include "detail/cuda_memory.hpp"
#include "detail/cuda_safe_call.hpp"

#define QMC_CORE_CUDA_SAFE_CALL(err) { integrators::core::cuda::detail::cuda_safe_call((err), __FILE__, __LINE__); }

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            template <typename F1, typename F2, typename T, typename D, typename U, typename G>
            void compute(
                         const Qmc<T, D, U, G>& qmc,
                         const U i, const U work_this_iteration, const U total_work_packages,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>>& d_z,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>>& d_d,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r,
                         const U d_r_size_over_m, const U n, const U m,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<F1>>& d_func,
                         const U dim,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<F2>>& d_integral_transform,
                         const int device
                         )
            {
                if (qmc.verbosity > 1) qmc.logger << "- (" << device << ") computing work_package " << i << ", work_this_iteration " << work_this_iteration << ", total_work_packages " << total_work_packages << std::endl;

                if(qmc.verbosity > 2) qmc.logger << "- (" << device << ") launching gpu kernel<<<" << qmc.cudablocks << "," << qmc.cudathreadsperblock << ">>>" << std::endl;

                integrators::core::cuda::compute_kernel<<< qmc.cudablocks, qmc.cudathreadsperblock >>>(i, work_this_iteration, total_work_packages,
                                                                                                       static_cast<U*>(*d_z),
                                                                                                       static_cast<D*>(*d_d),
                                                                                                       static_cast<T*>(*d_r),
                                                                                                       d_r_size_over_m, n, m,
                                                                                                       static_cast<F1*>(*d_func),
                                                                                                       dim,
                                                                                                       static_cast<F2*>(*d_integral_transform)
                                                                                                       );
                QMC_CORE_CUDA_SAFE_CALL(cudaDeviceSynchronize());

            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif

