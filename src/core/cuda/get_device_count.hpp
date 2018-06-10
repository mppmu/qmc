#ifndef QMC_CORE_CUDA_GET_DEVICE_COUNT_H
#define QMC_CORE_CUDA_GET_DEVICE_COUNT_H

#ifdef __CUDACC__
#include <cassert> // assert
#include <cuda_runtime_api.h> // cudaMemcpy

#include "detail/cuda_safe_call.hpp"

#define QMC_CORE_CUDA_SAFE_CALL(err) { integrators::core::cuda::detail::cuda_safe_call((err), __FILE__, __LINE__); }

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            int get_device_count()
            {
                int device_count;
                QMC_CORE_CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
                assert(device_count >= 0);
                return device_count;
            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif
