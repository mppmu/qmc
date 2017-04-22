#include <complex>
#include <cmath> // fabs, sqrt, isfinite

namespace integrators {

    template <typename T>
    std::complex<T> computeVariance(const std::complex<T>& mean, const std::complex<T>& variance, const std::complex<T>& sum, const std::complex<T>& delta )
    {
        return variance + std::complex<T>(delta.real()*(sum.real() - mean.real()), delta.imag()*(sum.imag() - mean.imag()));
    };
    
    template <typename T>
    std::complex<T> computeError(const std::complex<T>& svariance)
    {
        return std::complex<T>(std::sqrt(fabs(svariance.real())), std::sqrt(fabs(svariance.imag())));
    };

    template <typename T, typename D>
    bool computeIsFinite(const std::complex<T>& point, const D& wgt)
    {
        return ( std::isfinite(point.real()) && std::isfinite(point.imag()) && std::isfinite(wgt) );
    };
    
}
