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
            template <typename I, typename T, typename D>
            void setup_sample(
                                   std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>>& d_z, const std::vector<U>& z,
                                   std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>>& d_d, const std::vector<D>& d,
                                   std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r, const U d_r_size_over_m,
                                   const T* r_element, const U r_size_over_m, const U m,
                                   std::unique_ptr<integrators::core::cuda::detail::cuda_memory<I>>& d_func, I& func,
                                   const int device, const U verbosity, const Logger& logger
                             )
            {
                // Set Device
                if (verbosity > 1) logger << "- (" << device << ") setting device" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaSetDevice(device));
                if (verbosity > 1) logger << "- (" << device << ") device set" << std::endl;

                if(verbosity > 1) logger << "- (" << device << ") allocating d_func,d_z,d_d,d_r" << std::endl;
                d_func.reset( new integrators::core::cuda::detail::cuda_memory<I>(1) );
                d_z.reset( new integrators::core::cuda::detail::cuda_memory<U>(z.size()) );
                d_d.reset( new integrators::core::cuda::detail::cuda_memory<D>(d.size()) );
                d_r.reset( new integrators::core::cuda::detail::cuda_memory<T>(0,d_r_size_over_m*m) ); // allocate and set to 0
                if(verbosity > 1) logger << "- (" << device << ") allocated d_func,d_z,d_d,d_r" << std::endl;

                // copy func (initialize on new active device)
                if(verbosity > 1) logger << "- (" << device << ") initializing function on active device" << std::endl;
                I func_copy = func;
                if(verbosity > 1) logger << "- (" << device << ") initialized function on active device" << std::endl;

                if(verbosity > 1) logger << "- (" << device << ") copying d_func to device memory" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<I>::type*>(*d_func), &func_copy, sizeof(I), cudaMemcpyHostToDevice));
                if(verbosity > 1) logger << "- (" << device << ") copied d_func to device memory" << std::endl;

                // Copy z,d,r,func to device
                if(verbosity > 1) logger << "- (" << device << ") copying z,d to device memory" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<U*>(*d_z), z.data(), z.size() * sizeof(U), cudaMemcpyHostToDevice));
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<D*>(*d_d), d.data(), d.size() * sizeof(D), cudaMemcpyHostToDevice));
                // d_r not copied (initialised to 0 above)
                if(verbosity > 1) logger << "- (" << device << ") copied z,d to device memory" << std::endl;

                //        QMC_CORE_CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // TODO (V2) - investigate if this helps
                //        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, MyKernel, 0, 0); // TODO (V2) - investigate if this helps - https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
            };

            template <typename I, typename T, typename D>
            void setup_evaluate(
                                    std::unique_ptr<integrators::core::cuda::detail::cuda_memory<U>>& d_z, const std::vector<U>& z,
                                    std::unique_ptr<integrators::core::cuda::detail::cuda_memory<D>>& d_d, const std::vector<D>& d,
                                    std::unique_ptr<integrators::core::cuda::detail::cuda_memory<T>>& d_r, const U d_r_size,
                                    std::unique_ptr<integrators::core::cuda::detail::cuda_memory<I>>& d_func, I& func,
                                    const int device, const U verbosity, const Logger& logger
                                )
            {
                // Set Device
                if (verbosity > 1) logger << "- (" << device << ") setting device" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaSetDevice(device));
                if (verbosity > 1) logger << "- (" << device << ") device set" << std::endl;

                if(verbosity > 1) logger << "- (" << device << ") allocating d_func,d_z,d_d,d_r" << std::endl;
                d_func.reset( new integrators::core::cuda::detail::cuda_memory<I>(1) );
                d_z.reset( new integrators::core::cuda::detail::cuda_memory<U>(z.size()) );
                d_d.reset( new integrators::core::cuda::detail::cuda_memory<D>(d.size()) );
                d_r.reset( new integrators::core::cuda::detail::cuda_memory<T>(d_r_size) );
                if(verbosity > 1) logger << "- (" << device << ") allocated d_func,d_z,d_d,d_r" << std::endl;

                // copy func (initialize on new active device)
                if(verbosity > 1) logger << "- (" << device << ") initializing function on active device" << std::endl;
                I func_copy = func;
                if(verbosity > 1) logger << "- (" << device << ") initialized function on active device" << std::endl;

                if(verbosity > 1) logger << "- (" << device << ") copying d_func to device memory" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<I>::type*>(*d_func), &func_copy, sizeof(I), cudaMemcpyHostToDevice));
                if(verbosity > 1) logger << "- (" << device << ") copied d_func to device memory" << std::endl;

                // Copy z,d,func to device
                if(verbosity > 1) logger << "- (" << device << ") copying z,d,r to device memory" << std::endl;
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<U*>(*d_z), z.data(), z.size() * sizeof(U), cudaMemcpyHostToDevice));
                QMC_CORE_CUDA_SAFE_CALL(cudaMemcpy(static_cast<D*>(*d_d), d.data(), d.size() * sizeof(D), cudaMemcpyHostToDevice));
                if(verbosity > 1) logger << "- (" << device << ") copied z,d to device memory" << std::endl;

                //        QMC_CORE_CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // TODO (V2) - investigate if this helps
                //        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, MyKernel, 0, 0); // TODO (V2) - investigate if this helps - https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif
