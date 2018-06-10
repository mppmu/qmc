#ifndef QMC_OVERLOADS_REAL_H
#define QMC_OVERLOADS_REAL_H

#include <cmath> // abs, sqrt

namespace integrators
{
    namespace overloads
    {
        template <typename T>
        T compute_variance(const T& mean, const T& variance, const T& sum, const T& delta )
        {
            return variance + delta*(sum - mean);
        };

        template <typename T>
        T compute_error(const T& variance)
        {
            return T(std::sqrt(std::abs(variance)));
        };

        template <typename T>
        T compute_variance_from_error(const T& error)
        {
            return T(error*error);
        };

        template <typename T, typename D, typename U>
        D compute_error_ratio(const result<T,U>& res, const D& epsrel, const D&epsabs, const ErrorMode errormode)
        {
            return std::min(res.error/epsabs, std::abs(res.error/(res.integral*epsrel)));
        };
    };
};

#endif
