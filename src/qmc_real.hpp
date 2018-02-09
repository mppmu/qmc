#include <cmath> // abs, sqrt, isfinite

namespace integrators
{
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
    
    template <typename T, typename D, typename U>
    bool computeErrorGoalReached(const result<T,U>& res, const D& epsrel, const D& epsabs)
    {
        if (std::abs(res.error) <= std::max(std::abs(epsabs), std::abs(epsrel)*std::abs(res.integral)) )
            return true;
        return false;
    };
    
    template <typename T, typename D, typename U>
    D computeErrorRatio(const result<T,U>& res, const D& epsrel, const D&epsabs)
    {
        return std::max(res.error/epsabs, res.error*res.integral/epsrel);
    };
}
