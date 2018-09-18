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
            const std::vector<std::vector<D>> initial_parameters;

            D operator()(const D x, const double* p) const
            {
                throw std::logic_error("fit_function called");
            }
        };

        template<typename I, typename D, U maxdim>
        struct NoneTransform
        {
            static const U num_parameters = 0;

            I f; // original function
            const U dim;
            D p[maxdim][0]; // fit_parameters

            NoneTransform(const I& f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            }
        };

        template<typename I, typename D, U maxdim>
        struct NoneImpl
        {
            using function_t = NoneFunction<D>;
            using jacobian_t = std::nullptr_t;
            using hessian_t = std::nullptr_t;
            using transform_t = NoneTransform<I,D,maxdim>;
        };

        struct None
        {
            template<typename I, typename D, U maxdim> using type = NoneImpl<I, D, maxdim>;
        };
    };
};

#endif
