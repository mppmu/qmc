#include <cmath> // abs, sqrt, isfinite

namespace integrators {

    template <typename T>
    T computeVariance(const T& mean, const T& variance, const T& sum, const T& delta )
    {
        return variance + delta*(sum - mean);
    };
    
    template <typename T>
    T computeError(const T& variance)
    {
        return T(std::sqrt(std::abs(variance)));
    };

    template <typename T, typename D>
    bool computeIsFinite(const T& point, const D& wgt)
    {
        return ( std::isfinite(point) && std::isfinite(wgt) );
    };
    
}
