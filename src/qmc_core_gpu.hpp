#ifdef __CUDACC__
#include <iostream>
#include <stdexcept>
#include <exception>
#include <utility>
#include <string>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <memory> // unique_ptr
#include <cassert> // assert

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
    void compute_kernel_gpu(const U work_offset, const U work_this_iteration, const U total_work_packages, const U* z, const D* d, T* r, const U d_r_size_over_m, const U n, const U m, F1* func, const U dim, F2* integral_transform)
    {
        U i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < work_this_iteration)
        {
            for (U k = 0; k < m; k++)
            {
                T kahan_c = {0.};
                for (U offset = work_offset + i; offset < n; offset += total_work_packages)
                {
                    D wgt = 1.;
                    D mynull = 0;
                    D x[25]; // TODO - template parameter?

                    for (U sDim = 0; sDim < dim; sDim++)
                    {
                        x[sDim] = modf(integrators::mul_mod<D, D, U>(offset, z[sDim], n) / n + d[k*dim + sDim], &mynull);
                    }

                    (*integral_transform)(x, wgt, dim);

                    T point = (*func)(x);

                    // Compute sum using Kahan summation
                    // equivalent to: r[k*d_r_size_over_m + i] += wgt*point;
                    T kahan_y = wgt*point - kahan_c;
                    T kahan_t = r[k*d_r_size_over_m + i] + kahan_y;
                    T kahan_d = kahan_t - r[k*d_r_size_over_m + i];
                    kahan_c = kahan_d - kahan_y;
                    r[k*d_r_size_over_m + i] = kahan_t;
                }
            }
        }
    };

    template <typename F1, typename F2, typename T, typename D, typename U, typename G>
    void compute_gpu(
                        const Qmc<T, D, U, G>& qmc,
                        const U i, const U work_this_iteration, const U total_work_packages,
                        std::unique_ptr<integrators::detail::cuda_memory<U>>& d_z,
                        std::unique_ptr<integrators::detail::cuda_memory<D>>& d_d,
                        std::unique_ptr<integrators::detail::cuda_memory<T>>& d_r,
                        const U d_r_size_over_m, const U n, const U m,
                        std::unique_ptr<integrators::detail::cuda_memory<F1>>& d_func,
                        const U dim,
                        std::unique_ptr<integrators::detail::cuda_memory<F2>>& d_integral_transform,
                        const int device
                    )
    {
        if (qmc.verbosity > 1) qmc.logger << "- (" << device << ") computing work_package " << i << ", work_this_iteration " << work_this_iteration << ", total_work_packages " << total_work_packages << std::endl;

        if(qmc.verbosity > 2) qmc.logger << "- (" << device << ") launching gpu kernel<<<" << qmc.cudablocks << "," << qmc.cudathreadsperblock << ">>>" << std::endl;
        integrators::compute_kernel_gpu<<< qmc.cudablocks, qmc.cudathreadsperblock >>>(i, work_this_iteration, total_work_packages, static_cast<U*>(*d_z), static_cast<D*>(*d_d), static_cast<T*>(*d_r), d_r_size_over_m, n, m, static_cast<F1*>(*d_func), dim, static_cast<F2*>(*d_integral_transform));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    };

    template <typename F1, typename F2, typename T, typename D, typename U>
    void setup_gpu(
                        std::unique_ptr<integrators::detail::cuda_memory<U>>& d_z, const std::vector<U>& z,
                        std::unique_ptr<integrators::detail::cuda_memory<D>>& d_d, const std::vector<D>& d,
                        std::unique_ptr<integrators::detail::cuda_memory<T>>& d_r, const U d_r_size_over_m,
                        const T* r_element, const U r_size_over_m, const U m,
                        std::unique_ptr<integrators::detail::cuda_memory<F1>>& d_func, F1& func,
                        std::unique_ptr<integrators::detail::cuda_memory<F2>>& d_integral_transform, F2& integral_transform,
                        const int device, const U verbosity, const Logger& logger
                  )
    {
        // Set Device
        if (verbosity > 1) logger << "- (" << device << ") setting device" << std::endl;
        CUDA_SAFE_CALL(cudaSetDevice(device));
        if (verbosity > 1) logger << "- (" << device << ") device set" << std::endl;

        d_func.reset( new integrators::detail::cuda_memory<F1>(1) );
        d_integral_transform.reset( new integrators::detail::cuda_memory<F2>(1) );
        d_z.reset( new integrators::detail::cuda_memory<U>(z.size()) );
        d_d.reset( new integrators::detail::cuda_memory<D>(d.size()) );
        d_r.reset( new integrators::detail::cuda_memory<T>(d_r_size_over_m*m) );
        if(verbosity > 1) logger << "- (" << device << ") allocated d_func,d_integral_transform,d_z,d_d,d_r" << std::endl;

        // copy func and integral_transform (initialize on new active device)
        F1 func_copy = func;
        F2 integral_transform_copy = integral_transform;

        CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<F1>::type*>(*d_func), &func_copy, sizeof(F1), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(static_cast<typename std::remove_const<F2>::type*>(*d_integral_transform), &integral_transform_copy, sizeof(F2), cudaMemcpyHostToDevice));
        if(verbosity > 1) logger << "- (" << device << ") copied d_func,d_integral_transform to device memory" << std::endl;

        // Copy z,d,r,func,integral_transform to device
        CUDA_SAFE_CALL(cudaMemcpy(static_cast<U*>(*d_z), z.data(), z.size() * sizeof(U), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(static_cast<D*>(*d_d), d.data(), d.size() * sizeof(D), cudaMemcpyHostToDevice));
        for (U k = 0; k < m; k++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(&(static_cast<T*>(*d_r)[k*d_r_size_over_m]), &r_element[k*r_size_over_m], d_r_size_over_m * sizeof(T), cudaMemcpyHostToDevice));
        }

        if(verbosity > 1) logger << "- (" << device << ") copied z,d,r to device memory" << std::endl;

        //        CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // TODO - investigate if this helps
        //        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, MyKernel, 0, 0); // TODO - investigate if this helps - https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    };

    template <typename T, typename U>
    void tear_down_gpu(const std::unique_ptr<integrators::detail::cuda_memory<T>>& d_r, const U d_r_size_over_m, T* r_element, const U r_size_over_m, const U m, const int device, const U verbosity, const Logger& logger)
    {
        // Copy r to host
        for (U k = 0; k < m; k++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(&r_element[k*r_size_over_m], &(static_cast<T*>(*d_r)[k*d_r_size_over_m]), d_r_size_over_m * sizeof(T), cudaMemcpyDeviceToHost));
        }
        if (verbosity > 1) logger << "- (" << device << ") copied r to host memory" << std::endl;
    };

    int get_device_count_gpu()
    {
        int device_count;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
        assert(device_count >= 0);
        return device_count;
    };

};
#endif
