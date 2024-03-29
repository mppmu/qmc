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
            template <typename T, typename D, typename I>
            void compute(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U r_size_over_m, const U total_work_packages, const U n, const U m, const bool batching, I& func)
            {
                using std::modf;
                
                U batchsize = 1;
                T* points = nullptr;
                if (batching)
                {
                    batchsize = (n / total_work_packages) + ((i < (n % total_work_packages)) ? 1 : 0);
                    points = new T[batchsize];
                }
                std::vector<D> x(func.number_of_integration_variables * batchsize, 0);

                for (U k = 0; k < m; k++)
                {
                    for( U offset = i; offset < n; offset += total_work_packages )
                    {
                        D mynull = 0;

                        for (U sDim = 0; sDim < func.number_of_integration_variables; sDim++)
                        {
                            #define QMC_MODF_CALL modf( integrators::math::mul_mod<D,D>(offset,z.at(sDim),n)/(static_cast<D>(n)) + d.at(k*func.number_of_integration_variables+sDim), &mynull)

                            static_assert(std::is_same<decltype(QMC_MODF_CALL),D>::value, "Downcast detected in integrators::core::generic::compute. Please implement \"D modf(D)\".");
                            x[sDim + (batching ? (func.number_of_integration_variables * (offset / total_work_packages)) : 0)] = QMC_MODF_CALL;

                            #undef QMC_MODF_CALL
                        }
                        if (!batching) {
                            D wgt = 1.;
                            T point = func(x.data());
                            r_element[k*r_size_over_m] += wgt*point;
                        }
                    }

                    if constexpr (integrators::core::has_batching<I, T, D, U>) {
                        if (batching)
                        {
                            func(x.data(), points, batchsize);
                            D wgt = 1.;
                            for ( U i = 0; i != batchsize; ++i)
                            {
                                r_element[k*r_size_over_m] += wgt*points[i];
                            }
                        }
                    }
                }
                
                if (batching)
                    delete[] points;
            }

            template <typename T, typename D, typename I>
            void generate_samples(const U i, const std::vector<U>& z, const std::vector<D>& d, T* r_element, const U n, I& func)
            {
                using std::modf;

                D mynull = 0;
                std::vector<D> x(func.number_of_integration_variables,0);

                for (U sDim = 0; sDim < func.number_of_integration_variables; sDim++)
                {
                    #define QMC_MODF_CALL modf( integrators::math::mul_mod<D,D>(i,z.at(sDim),n)/(static_cast<D>(n)) + d.at(sDim), &mynull)

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

