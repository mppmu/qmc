#ifndef QMC_CORE_CUDA_TEARDOWN_H
#define QMC_CORE_CUDA_TEARDOWN_H

#ifdef __CUDACC__
#include <memory> // unique_ptr
#include <cuda_runtime_api.h> // cudaMemcpy

#include "detail/cuda_memory.hpp"
#include "detail/cuda_safe_call.hpp"

#define QMC_CORE_CUDA_SAFE_CALL(err) { integrators::core::cuda::detail::cuda_safe_call((err), __FILE__, __LINE__); }

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            template <typename T, typename U>
            void teardown_sample(const std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r, const U d_r_size_over_m, T* r_element, const U r_size_over_m, const U m, const int device, const U verbosity, const Logger& logger)
            {
                // Copy r to host
                for (U k = 0; k < m; k++)
                {
                    QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(&r_element[k*r_size_over_m], &(static_cast<T*>(*d_r)[k*d_r_size_over_m]), d_r_size_over_m * sizeof(T), cudaMemcpyDeviceToHost));
                }
                if (verbosity > 1) logger << "- (" << device << ") copied r to host memory" << std::endl;
            };

            template <typename T, typename U>
            void copy_back(const std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r, const U d_r_size, T* r_element, const int device, const U verbosity, const Logger& logger)
            {
                // Copy r_element to host
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(r_element, static_cast<T*>(*d_r), d_r_size * sizeof(T), cudaMemcpyDeviceToHost));
                if (verbosity > 1) logger << "- (" << device << ") copied r_element to host memory" << std::endl;
            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif
