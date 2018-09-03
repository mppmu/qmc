#ifndef QMC_FIT_TRANSFORM_H
#define QMC_FIT_TRANSFORM_H

#include <stdexcept> // std::domain_error

namespace integrators
{

    template <typename D>
    struct FitFunction
    {
        static const int num_parameters = 4;
        D operator()(const D x, const double* p) const
        {
            // constraint: no singularity
            if (p[0]>=static_cast<D>(0) && p[0]<=static_cast<D>(1))
                return std::numeric_limits<D>::max();

            D y = x*(p[1]+x*(p[2]+x*p[3])) + (D(1)-p[1]-p[2]-p[3])*(x*(p[0]-D(1)))/(p[0]-x);

            // constraint: transformed variable within unit hypercube
            if ( y<static_cast<D>(0) || y>static_cast<D>(1) )
                return std::numeric_limits<D>::max();
            
            return y;
        }
    };

    template <typename D>
    struct FitFunctionJacobian
    {
        static const int num_parameters = 4;
        D operator()(const D x, const double* p, const size_t parameter) const
        {
            if (parameter == 0) {
                return ((D(-1) + x)*x*(D(-1) + p[1] + p[2] + p[3]))/(x - p[0])/(x - p[0]);
            } else if (parameter == 1) {
                return x - (x*(D(-1) + p[0]))/(-x + p[0]);
            } else if (parameter == 2) {
                return x*x - (x*(D(-1) + p[0]))/(-x + p[0]);
            } else if (parameter == 3) {
                return x*x*x - (x*(D(-1) + p[0]))/(-x + p[0]);
            } else {
                throw std::domain_error("fit_function_jacobian called with invalid parameter: " + std::to_string(parameter));
            }
        }
    };

    template<typename F1, typename D, typename U = unsigned long long int, U maxdim = 25>
    struct FitTransform {
        const static U num_params = 4;

        F1 f; // original function
        const U dim;
        D p[maxdim][num_params]; // fit_parameters

        FitTransform(const F1& f) : f(f), dim(f.dim) {};

#ifdef __CUDACC__
        __host__ __device__
#endif
        auto operator()(D* x) -> decltype(f(x)) const
        {
            D wgt = 1;
            for (U d = 0; d < dim ; ++d)
            {
                D q = D(1)-p[d][1]-p[d][2]-p[d][3];
                wgt *= p[d][1] + x[d]*(D(2)*p[d][2]+x[d]*D(3)*p[d][3]) + q*p[d][0]*(p[d][0]-D(1))/(p[d][0]-x[d])/(p[d][0]-x[d]);
                x[d] = x[d]*(p[d][1]+x[d]*(p[d][2]+x[d]*p[d][3])) + q*x[d]*(p[d][0]-D(1))/(p[d][0]-x[d]);
                if ( x[d] > D(1) || x[d] < D(0) ) return D(0);
            }
            return wgt * f(x);
        }
    };

};

#endif
