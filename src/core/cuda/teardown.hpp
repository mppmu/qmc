#ifndef QMC_CORE_CUDA_TEARDOWN_H
#define QMC_CORE_CUDA_TEARDOWN_H

#ifdef __CUDACC__
#include <memory> // unique_ptr
#include <cuda_runtime_api.h> // cudaMemcpy
#include <thrust/device_ptr.h> // thrust::device_ptr
#include <thrust/reduce.h> // thrust::reduce

#include "detail/cuda_memory.hpp"
#include "detail/cuda_safe_call.hpp"

#define QMC_CORE_CUDA_SAFE_CALL(err) { integrators::core::cuda::detail::cuda_safe_call((err), __FILE__, __LINE__); }

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            template <typename T>
            void teardown_sample(const std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r, const U d_r_size_over_m, T* r_element, const U r_size_over_m, const U m, const int device, const U verbosity, const Logger& logger)
            {
                // Wrap raw device pointer into a thrust::device_ptr
                thrust::device_ptr<T> d_r_ptr(static_cast<T*>(*d_r.get()));
                // Reduce d_r on device and copy result to host
                for (U k = 0; k < m; k++)
                {
                    r_element[k*r_size_over_m] = thrust::reduce(d_r_ptr+k*d_r_size_over_m, d_r_ptr+k*d_r_size_over_m + d_r_size_over_m);
                }
                if (verbosity > 1) logger << "- (" << device << ") copied r to host memory" << std::endl;
            };

            template <typename T>
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
