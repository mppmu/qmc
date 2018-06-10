#ifndef QMC_CORE_CUDA_DETAIL_CUDA_SAFE_CALL_H
#define QMC_CORE_CUDA_DETAIL_CUDA_SAFE_CALL_H

#ifdef __CUDACC__
#include <stdexcept>
#include <exception>
#include <string>

#include <cuda_runtime_api.h>

namespace integrators
{
    namespace core
    {
        namespace cuda
        {
            namespace detail
            {
                struct cuda_error : public std::runtime_error { using std::runtime_error::runtime_error; };

                inline void cuda_safe_call(cudaError_t error, const char *file, int line)
                {
                    if (error != cudaSuccess)
                    {
                        throw cuda_error(std::string(cudaGetErrorString(error)) + ": " + std::string(file) + " line " + std::to_string(line));
                    }
                };
            };
        };
    };
};

#endif
#endif
