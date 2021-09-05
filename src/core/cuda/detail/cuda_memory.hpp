#ifndef QMC_CORE_CUDA_DETAIL_CUDA_MEMORY_H
#define QMC_CORE_CUDA_DETAIL_CUDA_MEMORY_H

#ifdef __CUDACC__
#include <type_traits> // remove_const
#include <cuda_runtime_api.h>

#include "cuda_safe_call.hpp"

#define QMC_CORE_CUDA_SAFE_CALL(err) { integrators::core::cuda::detail::cuda_safe_call((err), __FILE__, __LINE__); }

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            namespace detail
            {
                template<typename Tin>
                class cuda_memory
                {
                public:
                    using T = typename std::remove_const<Tin>::type;
                private:
                    T* memory;
                public:
                    operator T*() { return memory; }
                    cuda_memory(std::size_t s) { QMC_CORE_CUDA_SAFE_CALL(cudaMalloc(&memory, s*sizeof(T))); };
                    cuda_memory(int value, std::size_t s) { 
                        QMC_CORE_CUDA_SAFE_CALL(cudaMalloc(&memory, s*sizeof(T)));
                        QMC_CORE_CUDA_SAFE_CALL(cudaMemset(memory, value, s*sizeof(T)));
                    };
                    ~cuda_memory() { QMC_CORE_CUDA_SAFE_CALL(cudaFree(memory)); }
                };
            };
        };
    };
};

#undef QMC_CORE_CUDA_SAFE_CALL

#endif
#endif
