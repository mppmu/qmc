#include <complex>
#include <cmath> // abs, sqrt, isfinite

#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

namespace integrators
{
    // Implementation
    template <typename T>
    T computeVariance_complex(const T& mean, const T& variance, const T& sum, const T& delta )
    {
        return variance + T(delta.real()*(sum.real() - mean.real()), delta.imag()*(sum.imag() - mean.imag()));
    }
    
    template <typename T>
    T computeError_complex(const T& svariance)
    {
        return T(std::sqrt(std::abs(svariance.real())), std::sqrt(std::abs(svariance.imag())));
    };
    
    template <typename T, typename D>
    bool computeIsFinite_complex(const T& point, const D& wgt)
    {
        return ( std::isfinite(point.real()) && std::isfinite(point.imag()) && std::isfinite(wgt) );
    };
    
    template <typename T, typename D, typename U>
    bool computeErrorGoalReached_complex(const result<T,U>& res, const D& epsrel, const D& epsabs)
    {
        if ( ( std::abs(res.error.real()) <= std::max(std::abs(epsabs), std::abs(epsrel)*std::abs(res.integral.real())) )
            &&
            ( std::abs(res.error.imag()) <= std::max(std::abs(epsabs), std::abs(epsrel)*std::abs(res.integral.imag())) ) )
            return true;
        return false;
    };
    
    template <typename T, typename D, typename U>
    D computeErrorRatio_complex(const result<T,U>& res, const D& epsrel, const D& epsabs)
    {
        return std::max(
                        std::max(res.error.real(),res.error.imag())/epsabs,
                        std::max(res.error.real(),res.error.imag())*std::max(res.integral.real(),res.integral.imag())/epsrel
                        );
    };
    
    // Overloads (std::complex)
    template <typename T> std::complex<T> computeVariance(const std::complex<T>& mean, const std::complex<T>& variance, const std::complex<T>& sum, const std::complex<T>& delta ) { return computeVariance_complex(mean,variance,sum,delta); };
    template <typename T> std::complex<T> computeError(const std::complex<T>& svariance) { return computeError_complex(svariance); };
    template <typename T, typename D> bool computeIsFinite(const std::complex<T>& point, const D& wgt) { return computeIsFinite_complex(point,wgt); };
    template <typename T, typename D, typename U> bool computeErrorGoalReached(const result<std::complex<T>,U>& res, const D& epsrel, const D& epsabs) { return computeErrorGoalReached_complex(res, epsrel, epsabs); };
    template <typename T, typename D, typename U> D computeErrorRatio(const result<std::complex<T>,U>& res, const D& epsrel, const D& epsabs) { return computeErrorRatio_complex(res, epsrel, epsabs); };
    
#ifdef __CUDACC__
    // Overloads (thrust::complex)
    template <typename T> thrust::complex<T> computeVariance(const thrust::complex<T>& mean, const thrust::complex<T>& variance, const thrust::complex<T>& sum, const thrust::complex<T>& delta ) { return computeVariance_complex(mean,variance,sum,delta); };
    template <typename T> thrust::complex<T> computeError(const thrust::complex<T>& svariance) { return computeError_complex(svariance); };
    template <typename T, typename D> bool computeIsFinite(const thrust::complex<T>& point, const D& wgt) { return computeIsFinite_complex(point,wgt); };
    template <typename T, typename D, typename U> bool computeErrorGoalReached(const result<thrust::complex<T>,U>& res, const D& epsrel, const D& epsabs) { return computeErrorGoalReached_complex(res, epsrel, epsabs); };
    template <typename T, typename D, typename U> D computeErrorRatio(const result<thrust::complex<T>,U>& res, const D& epsrel, const D& epsabs) { return computeErrorRatio_complex(res, epsrel, epsabs); };
#endif
}

