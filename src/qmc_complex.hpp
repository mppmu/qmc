#include <complex>
#include <cmath> // abs, sqrt

#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

namespace integrators
{
    // Implementation
    template <typename T>
    T compute_variance_complex(const T& mean, const T& variance, const T& sum, const T& delta )
    {
        return variance + T(delta.real()*(sum.real() - mean.real()), delta.imag()*(sum.imag() - mean.imag()));
    }
    
    template <typename T>
    T compute_error_complex(const T& svariance)
    {
        return T(std::sqrt(std::abs(svariance.real())), std::sqrt(std::abs(svariance.imag())));
    };
    
    template <typename T, typename D, typename U>
    D compute_error_ratio_complex(const result<T,U>& res, const D& epsrel, const D& epsabs)
    {
        return std::max(
                        std::min(res.error.real()/epsabs, res.error.real()/std::abs(res.integral.real()*epsrel)),
                        std::min(res.error.imag()/epsabs, res.error.imag()/std::abs(res.integral.imag()*epsrel))
                        );
    };
    
    template <typename T, typename D, typename U>
    D compute_error_ratio_complex_modified(const result<T,U>& res, const D& epsrel, const D& epsabs)
    {
        return std::min(
                        std::max(res.error.real(),res.error.imag())/epsabs,
                        std::max(res.error.real(),res.error.imag())/(std::max(std::abs(res.integral.real()),std::abs(res.integral.imag()))*epsrel)
                        );
    };
    
    // Overloads (std::complex)
    template <typename T> std::complex<T> compute_variance(const std::complex<T>& mean, const std::complex<T>& variance, const std::complex<T>& sum, const std::complex<T>& delta ) { return compute_variance_complex(mean,variance,sum,delta); };
    template <typename T> std::complex<T> compute_error(const std::complex<T>& svariance) { return compute_error_complex(svariance); };
    template <typename T, typename D, typename U> D compute_error_ratio(const result<std::complex<T>,U>& res, const D& epsrel, const D& epsabs) { return compute_error_ratio_complex(res, epsrel, epsabs); };
    
#ifdef __CUDACC__
    // Overloads (thrust::complex)
    template <typename T> thrust::complex<T> compute_variance(const thrust::complex<T>& mean, const thrust::complex<T>& variance, const thrust::complex<T>& sum, const thrust::complex<T>& delta ) { return compute_variance_complex(mean,variance,sum,delta); };
    template <typename T> thrust::complex<T> compute_error(const thrust::complex<T>& svariance) { return compute_error_complex(svariance); };
    template <typename T, typename D, typename U> D compute_error_ratio(const result<thrust::complex<T>,U>& res, const D& epsrel, const D& epsabs) { return compute_error_ratio_complex(res, epsrel, epsabs); };
#endif
}

