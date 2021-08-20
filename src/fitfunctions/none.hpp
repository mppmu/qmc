#ifndef QMC_FITFUNCTIONS_NONE_H
#define QMC_FITFUNCTIONS_NONE_H

#include <cstddef> // nullptr_t
#include <stdexcept> // logic_error

namespace integrators
{
    namespace fitfunctions
    {

        template <typename D>
        struct NoneFunction
        {
            static const int num_parameters = 0;
            const std::vector<std::vector<D>> initial_parameters = {};

            D operator()(const D x, const double* p) const
            {
                throw std::logic_error("fit_function called");
            }
        };
        template <typename T, typename DP, typename RP, typename U> static constexpr bool hasEvaluate(...) {
            return false;
        }

        template <typename T, typename DP, typename RP, typename U> static constexpr bool hasEvaluate(int, decltype((std::declval<T>().evaluate(DP(), RP(), U()))) = U()) {
            return true;
        }

        template<typename I, typename D, U M>
        struct NoneTransform
        {
            static const U num_parameters = 0;

            I f; // original function
            const U number_of_integration_variables;
            D p[M][1]; // fit_parameters

            NoneTransform(const I& f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x))
            {
                return f(x);
            }
            void evaluate(D* x, decltype(f(x))* res, U count)
            {
                if constexpr (hasEvaluate<I, D*, decltype(f(x))*, U>(0)) {
                    f.evaluate(x, res, count);
                } else {
                    for (U i = U(); i != count; ++i) {
                        res[i] = f(x + i * f.number_of_integration_variables);
                    }
                }
            }
        };

        template<typename I, typename D, U M>
        struct NoneImpl
        {
            using function_t = NoneFunction<D>;
            using jacobian_t = std::nullptr_t;
            using hessian_t = std::nullptr_t;
            using transform_t = NoneTransform<I,D,M>;
        };

        struct None
        {
            template<typename I, typename D, U M> using type = NoneImpl<I, D, M>;
        };
    };
};

#endif
