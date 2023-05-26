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
            using std::sqrt;
            using std::abs;
            return T(sqrt(abs(variance)));
        };

        template <typename T>
        T compute_variance_from_error(const T& error)
        {
            return T(error*error);
        };

        template <typename T, typename D>
        D compute_error_ratio(const result<T>& res, const D& epsrel, const D&epsabs, const ErrorMode errormode)
        {
            using std::abs;

            #define QMC_ABS_CALL abs(res.error/(res.integral*epsrel))

            static_assert(std::is_same<decltype(QMC_ABS_CALL),D>::value, "Downcast detected in integrators::overloads::compute_error_ratio. Please implement \"D abs(D)\".");
            return std::min(res.error/epsabs, QMC_ABS_CALL);

            #undef QMC_ABS_CALL
        };

        template <typename T>
        T compute_signed_max_re_im(const result<T>& res)
        {
            return res.integral;
        };
    };
};

#endif
