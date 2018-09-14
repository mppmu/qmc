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
            template <typename I, typename T, typename D, typename Q>
            void compute(
                         const Q& qmc,
                         const U i, const U work_this_iteration, const U total_work_packages,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>>& d_z,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>>& d_d,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r,
                         const U d_r_size_over_m, const U n, const U m,
                         std::unique_ptr<integrators::core::cuda::detail::cuda_memory<I>>& d_func,
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
                                                                                                       static_cast<I*>(*d_func)
                                                                                                       );
                QMC_CORE_CUDA_SAFE_CALL(cudaDeviceSynchronize());

            };

            template <typename I, typename T, typename D, typename Q>
            void generate_samples(
                                     const Q& qmc,
                                     const U i_start, const U work_this_iteration,
                                     std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>>& d_z,
                                     std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>>& d_d,
                                     std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r,
                                     const U n,
                                     std::unique_ptr<integrators::core::cuda::detail::cuda_memory<I>>& d_func,
                                     const int device
                                 )
            {
                if (qmc.verbosity > 1) qmc.logger << "- (" << device << ") computing samples " << i_start << ", work_this_iteration " << work_this_iteration << ", n " << n << std::endl;

                if(qmc.verbosity > 2) qmc.logger << "- (" << device << ") launching gpu kernel<<<" << qmc.cudablocks << "," << qmc.cudathreadsperblock << ">>>" << std::endl;

                integrators::core::cuda::generate_samples_kernel<<< qmc.cudablocks, qmc.cudathreadsperblock >>>(i_start, work_this_iteration,
                                                                                                                static_cast<U*>(*d_z),
                                                                                                                static_cast<D*>(*d_d),
                                                                                                                static_cast<T*>(*d_r),
                                                                                                                n,
                                                                                                                static_cast<I*>(*d_func)
                                                                                                                );
                QMC_CORE_CUDA_SAFE_CALL(cudaDeviceSynchronize());

            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif

