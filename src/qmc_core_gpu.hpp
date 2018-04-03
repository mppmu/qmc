#ifdef __CUDACC__
#include <stdexcept>
#include <exception>
#include <utility>
#include <string>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL(err) { integrators::detail::cuda_safe_call_impl((err), __FILE__, __LINE__); }

namespace integrators
{
    struct cuda_error : public std::runtime_error { using std::runtime_error::runtime_error; };

    namespace detail
    {
        inline void cuda_safe_call_impl(cudaError_t error, const char *file, int line)
        {
            if (error != cudaSuccess)
            {
                throw cuda_error(std::string(cudaGetErrorString(error)) + ": " + std::string(file) + " line " + std::to_string(line));
            }
        };

        template<typename Tin>
        class cuda_memory
        {
        public:
            using T = typename std::remove_const<Tin>::type;
        private:
            T* memory;
        public:
            operator T*() { return memory; }
            cuda_memory(std::size_t s) { CUDA_SAFE_CALL(cudaMalloc(&memory, s*sizeof(T))); };
            ~cuda_memory() { CUDA_SAFE_CALL(cudaFree(memory)); }
        };

    };

    // TODO - make use of restricted pointers?
    template <typename T, typename D, typename U, typename F1, typename F2>
    __global__
    void compute_kernel_gpu(const U work_offset, const U points_per_package, const U work_this_iteration, const U total_work_packages, const U* z, const D* d, T* r, const U n, const U m, F1* func, const U dim, F2* integralTransform, const D border)
    {
        U i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < work_this_iteration)
        {
            for (U k = 0; k < m; k++)
            {
                for (U b = 0; b < points_per_package; b++)
                {
                    U offset = b * total_work_packages + work_offset;
                    if (offset + i < n)
                    {
                        D wgt = 1.;
                        D mynull = 0;
                        D x[25]; // TODO - template parameter?

                        for (U sDim = 0; sDim < dim; sDim++)
                        {
                            x[sDim] = modf(integrators::mul_mod<D, D, U>(i + offset, z[sDim], n) / n + d[k*dim + sDim], &mynull);
                        }

                        (*integralTransform)(x, wgt, dim);

                        // Nudge point inside border (for numerical stability)
                        for (U sDim = 0; sDim < dim; sDim++)
                        {
                            if( x[sDim] < border)
                                x[sDim] = border;
                            if( x[sDim] > 1.-border)
                                x[sDim] = 1.-border;
                        }

                        T point = (*func)(x);

                        r[k*work_this_iteration + i] += wgt*point; // TODO - Compute sum using Kahan summation?
                    }
                }
            }
        }
    };

    template <typename T, typename D, typename U, typename G>
    template <typename F1, typename F2>
    void Qmc<T, D, U, G>::compute_gpu(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size, const U work_this_iteration, const U total_work_packages, const U points_per_package, const U n, const U m, F1& func, const U dim, F2& integralTransform, const int device, const U cudablocks, const U cudathreadsperblock)
    {
        if (verbosity > 1) std::cout << "- (" << device << ") computing work_package " << i << ", work_this_iteration " << work_this_iteration << ", total_work_packages " << total_work_packages << std::endl;
        // Set Device
        if (verbosity > 1) std::cout << "- (" << device << ") setting device" << std::endl;
        CUDA_SAFE_CALL(cudaSetDevice(device));
        if (verbosity > 1) std::cout << "- (" << device << ") device set" << std::endl;

        // copy func and integralTransform (initialize on new active device)
        F1 func_copy = func;
        F2 integralTransform_copy = integralTransform;

        // Allocate Device Memory
        integrators::detail::cuda_memory<U> d_z(z.size());
        integrators::detail::cuda_memory<D> d_d(d.size());
        integrators::detail::cuda_memory<T> d_r(m*work_this_iteration);
        integrators::detail::cuda_memory<F1> d_func(1); // TODO: copy only once per device
        integrators::detail::cuda_memory<F2> d_integralTransform(1); // TODO: copy only once per device
        if (verbosity > 1) std::cout << "- (" << device << ") allocated device memory" << std::endl;

        
        //        CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // TODO - investigate if this helps
        //        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, MyKernel, 0, 0); // TODO - investigate if this helps - https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
        // Copy z,d,r,func,integralTransform to device
        CUDA_SAFE_CALL(cudaMemcpy(d_z, z.data(), z.size() * sizeof(U), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_d, d.data(), d.size() * sizeof(D), cudaMemcpyHostToDevice));
        for (U k = 0; k < m; k++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(&(static_cast<T*>(d_r)[k*work_this_iteration]), &r_element[k*r_size], work_this_iteration * sizeof(T), cudaMemcpyHostToDevice));
        }
        CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<F1>::type*>(d_func), &func_copy, sizeof(F1), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<F2>::type*>(d_integralTransform), &integralTransform_copy, sizeof(F2), cudaMemcpyHostToDevice));

        if(verbosity > 1) std::cout << "- (" << device << ") copied z,d,r to device memory" << std::endl;
        if(verbosity > 1) std::cout << "- (" << device << ") allocated d_z " << z.size() << std::endl;
        if(verbosity > 1) std::cout << "- (" << device << ") allocated d_d " << d.size() << std::endl;
        if(verbosity > 1) std::cout << "- (" << device << ") allocated d_r " << m*work_this_iteration << std::endl;
        if(verbosity > 2) std::cout << "- (" << device << ") launching gpu kernel<<<" << cudablocks << "," << cudathreadsperblock << ">>>" << std::endl;

        integrators::compute_kernel_gpu<<< cudablocks, cudathreadsperblock >>>(i,points_per_package, work_this_iteration, total_work_packages, static_cast<U*>(d_z), static_cast<D*>(d_d), static_cast<T*>(d_r), n, m, static_cast<F1*>(d_func), dim, static_cast<F2*>(d_integralTransform), border);
//        CUDA_SAFE_CALL(cudaPeekAtLastError());
//        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // Copy r to host
        for (U k = 0; k < m; k++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(&r_element[k*r_size], &(static_cast<T*>(d_r)[k*work_this_iteration]), work_this_iteration * sizeof(T), cudaMemcpyDeviceToHost));
        }
        if (verbosity > 1) std::cout << "- (" << device << ") copied r to host memory" << std::endl;
    };
};
#endif
