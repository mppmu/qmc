#ifndef QMC_CORE_CUDA_SETUP_H
#define QMC_CORE_CUDA_SETUP_H

#ifdef __CUDACC__
#include <memory> // unique_ptr
#include <cassert> // assert
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
            template <typename F1, typename F2, typename T, typename D, typename U>
            void setup(
                           std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>>& d_z, const std::vector<U>& z,
                           std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>>& d_d, const std::vector<D>& d,
                           std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r, const U d_r_size_over_m,
                           const T* r_element, const U r_size_over_m, const U m,
                           std::unique_ptr<integrators::core::cuda::detail::cuda_memory<F1>>& d_func, F1& func,
                           std::unique_ptr<integrators::core::cuda::detail::cuda_memory<F2>>& d_integral_transform, F2& integral_transform,
                           const int device, const U verbosity, const Logger& logger
                           )
            {
                // Set Device
                if (verbosity > 1) logger << "- (" << device << ") setting device" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaSetDevice(device));
                if (verbosity > 1) logger << "- (" << device << ") device set" << std::endl;

                d_func.reset( new integrators::core::cuda::detail::cuda_memory<F1>(1) );
                d_integral_transform.reset( new integrators::core::cuda::detail::cuda_memory<F2>(1) );
                d_z.reset( new integrators::core::cuda::detail::cuda_memory<U>(z.size()) );
                d_d.reset( new integrators::core::cuda::detail::cuda_memory<D>(d.size()) );
                d_r.reset( new integrators::core::cuda::detail::cuda_memory<T>(d_r_size_over_m*m) );
                if(verbosity > 1) logger << "- (" << device << ") allocated d_func,d_integral_transform,d_z,d_d,d_r" << std::endl;

                // copy func and integral_transform (initialize on new active device)
                F1 func_copy = func;
                F2 integral_transform_copy = integral_transform;

                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<F1>::type*>(*d_func), &func_copy, sizeof(F1), cudaMemcpyHostToDevice));
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<F2>::type*>(*d_integral_transform), &integral_transform_copy, sizeof(F2), cudaMemcpyHostToDevice));
                if(verbosity > 1) logger << "- (" << device << ") copied d_func,d_integral_transform to device memory" << std::endl;

                // Copy z,d,r,func,integral_transform to device
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<U*>(*d_z), z.data(), z.size() * sizeof(U), cudaMemcpyHostToDevice));
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<D*>(*d_d), d.data(), d.size() * sizeof(D), cudaMemcpyHostToDevice));
                for (U k = 0; k < m; k++)
                {
                    QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(&(static_cast<T*>(*d_r)[k*d_r_size_over_m]), &r_element[k*r_size_over_m], d_r_size_over_m * sizeof(T), cudaMemcpyHostToDevice));
                }

                if(verbosity > 1) logger << "- (" << device << ") copied z,d,r to device memory" << std::endl;

                //        QMC_CORE_CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // TODO - investigate if this helps
                //        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, MyKernel, 0, 0); // TODO - investigate if this helps - https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif
