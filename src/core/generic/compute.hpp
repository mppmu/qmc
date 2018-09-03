#ifndef QMC_CORE_GENERIC_COMPUTE_H
#define QMC_CORE_GENERIC_COMPUTE_H

#include <cmath>

#include "../../math/mul_mod.hpp"

namespace integrators
{
    namespace core
    {
        namespace generic
        {
            template <typename T, typename D, typename U, typename F1>
            void compute(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size_over_m, const U total_work_packages, const U n, const U m, F1& func)
            {
                using std::modf;

                for (U k = 0; k < m; k++)
                {
                    T kahan_c = {0.};
                    for( U offset = i; offset < n; offset += total_work_packages )
                    {
                        D wgt = 1.;
                        D mynull = 0;
                        std::vector<D> x(func.dim,0);

                        for (U sDim = 0; sDim < func.dim; sDim++)
                        {
                            #define QMC_MODF_CALL modf( integrators::math::mul_mod<D,D,U>(offset,z.at(sDim),n)/(static_cast<D>(n)) + d.at(k*func.dim+sDim), &mynull)

                            static_assert(std::is_same<decltype(QMC_MODF_CALL),D>::value, "Downcast detected in integrators::core::generic::compute. Please implement \"D modf(D)\".");
                            x[sDim] = QMC_MODF_CALL;

                            #undef QMC_MODF_CALL
                        }

                        T point = func(x.data());

                        // Compute sum using Kahan summation
                        // equivalent to: r_element[k*r_size_over_m] += wgt*point;
                        T kahan_y = wgt*point - kahan_c;
                        T kahan_t = r_element[k*r_size_over_m] + kahan_y;
                        T kahan_d = kahan_t - r_element[k*r_size_over_m];
                        kahan_c = kahan_d - kahan_y;
                        r_element[k*r_size_over_m] = kahan_t;
                    }
                }
            }

            template <typename T, typename D, typename U, typename F1>
            void generate_samples(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U n, F1& func)
            {
                using std::modf;

                D mynull = 0;
                std::vector<D> x(func.dim,0);

                for (U sDim = 0; sDim < func.dim; sDim++)
                {
                    #define QMC_MODF_CALL modf( integrators::math::mul_mod<D,D,U>(i,z.at(sDim),n)/(static_cast<D>(n)) + d.at(sDim), &mynull)

                    static_assert(std::is_same<decltype(QMC_MODF_CALL),D>::value, "Downcast detected in integrators::core::generic::compute. Please implement \"D modf(D)\".");
                    x[sDim] = QMC_MODF_CALL;

                    #undef QMC_MODF_CALL
                }

                *r_element = func(x.data());
            }
        };
    };
};

#endif

