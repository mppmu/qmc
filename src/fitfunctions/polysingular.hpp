#ifndef QMC_FITFUNCTIONS_POLYSINGULAR_H
#define QMC_FITFUNCTIONS_POLYSINGULAR_H

#include <stdexcept> // std::domain_error
#include <vector>
#include <cmath>

#include <iostream>

namespace integrators
{
    namespace fitfunctions
    {
        template <typename D>
        struct PolySingularFunction
        {
            static const int num_parameters = 6;
            const std::vector<std::vector<D>> initial_parameters = { {1.1,-0.1, 0.1,0.1, 0.3,0.3} };

            D operator()(const D x, const double* p) const
            {
                // constraint: no singularity and singular terms have positive coefficients
                if (p[0]<=static_cast<D>(1.001) or p[1]>=static_cast<D>(-0.001) )
                    return D(10.); // std::numeric_limits<D>::max() will sometimes result in fit parameters being NaN
//                if (p[0]<=static_cast<D>(1.001) or p[0]>=static_cast<D>(5) or p[1]>=static_cast<D>(-0.001) or p[1]<=static_cast<D>(-4) )
//                //if (p[0]<=static_cast<D>(1.001) or p[0]>=static_cast<D>(1.5) or p[1]>=static_cast<D>(-0.001) or p[1]<=static_cast<D>(-0.5) )
//                    return D(10.); // std::numeric_limits<D>::max() will sometimes result in fit parameters being NaN
                
                D p2 = abs(p[2]);
                D p3 = abs(p[3]);
                D y = p2*(x*(p[0]-D(1)))/(p[0]-x) + p3*(x*(p[1]-D(1)))/(p[1]-x)  + x*(p[4]+x*(p[5]+x*(D(1)-p2-p3-p[4]-p[5])));

                // constraint: transformed variable within unit hypercube
                if ( y<static_cast<D>(0) || y>static_cast<D>(1) )
                    return std::numeric_limits<D>::max();

                return y;
            }
        };

        template <typename D>
        struct PolySingularRegularization
        {
            static const int num_parameters = 6;

            D operator()(const double* p, const size_t parameter, const bool jacobian) const
            {
                if (parameter == 0) {
                    if (p[parameter] < 1.3) return D(0);
                    return (jacobian) ? D(1) : (p[parameter]-D(1.3));
                } else if (parameter == 1) {
                    if (p[parameter] > -0.3) return D(0);
                    return (jacobian) ? D(-1) : (D(-0.3)-p[parameter]);
                } else if (parameter < 6) {
                    if (abs(p[parameter]) < 2) return D(0);
                    return (jacobian) ? copysign(1.,p[parameter]) : (abs(p[parameter])-2);
                } else {
                    throw std::domain_error("fit_function_regularization called with invalid parameter: " + std::to_string(parameter));
                }
            }
        };

        template <typename D>
        struct PolySingularJacobian
        {
            static const int num_parameters = 6;

            D operator()(const D x, const double* p, const size_t parameter) const
            {

                if (parameter == 0) {
                    return abs(p[2])*((D(1) - x)*x)/(x - p[0])/(x - p[0]);
                } else if (parameter == 1) {
                    return abs(p[3])*((D(1) - x)*x)/(x - p[1])/(x - p[1]);
                } else if (parameter == 2) {
                    return ((x*(p[0]-D(1)))/(p[0]-x) -x*x*x) * copysign(1.,p[2]);
                } else if (parameter == 3) {
                    return ((x*(p[1]-D(1)))/(p[1]-x) -x*x*x) * copysign(1.,p[3]);
                } else if (parameter == 4) {
                    return  x*(D(1)-x*x);
                } else if (parameter == 5) {
                    return  x*x*(D(1)-x);
                } else {
                    throw std::domain_error("fit_function_jacobian called with invalid parameter: " + std::to_string(parameter));
                }
            }
        };
        template <typename D>
        struct PolySingularHessian
        {
            D operator()(const D x, const double* v, const double* p) const
            {
                D xmp0 = x-p[0];
                D xmp1 = x-p[1];
                return x*(D(1)-x)*D(2)*(v[0]*(abs(p[2])*v[0]+(x - p[0])*v[2])/xmp0/xmp0/xmp0 + v[1]*(abs(p[3])*v[1]+(x - p[1])*v[3])/xmp1/xmp1/xmp1);
          }
        };

        template<typename I, typename D, U M>
        struct PolySingularTransform
        {
            static const U num_parameters = 6;

            I f; // original function
            const U number_of_integration_variables;
            D p[M][num_parameters]; // fit_parameters

            PolySingularTransform(const I& f) : f(f), number_of_integration_variables(f.number_of_integration_variables) {}

#ifdef __CUDACC__
            __host__ __device__
#endif
            auto operator()(D* x) -> decltype(f(x)) const
            {
                D wgt = 1;
                for (U d = 0; d < number_of_integration_variables ; ++d)
                {
                    D p2 = abs(p[d][2]);
                    D p3 = abs(p[d][3]);
                    wgt *= p2*p[d][0]*(p[d][0]-D(1))/(p[d][0]-x[d])/(p[d][0]-x[d]) + p3*p[d][1]*(p[d][1]-D(1))/(p[d][1]-x[d])/(p[d][1]-x[d]) + p[d][4] + x[d]*(D(2)*p[d][5]+x[d]*D(3)*(D(1)-p2-p3-p[d][4]-p[d][5]));
                    x[d] = p2*(x[d]*(p[d][0]-D(1)))/(p[d][0]-x[d]) + p3*(x[d]*(p[d][1]-D(1)))/(p[d][1]-x[d])  + x[d]*(p[d][4]+x[d]*(p[d][5]+x[d]*(D(1)-p2-p3-p[d][4]-p[d][5])));
                    if ( x[d] > D(1) || x[d] < D(0) ) return D(0);
                }
                return wgt * f(x);
            }
        };

        template<typename I, typename D, U M>
        struct PolySingularImpl
        {
            using function_t = PolySingularFunction<D>;
            //using jacobian_t = std::nullptr_t; //PolySingularJacobian<D>; // set to std::nullptr_t to compute numerically
            //using hessian_t = std::nullptr_t; //PolySingularHessian<D>; // set to std::nullptr_t to compute numerically (also set fitparametersgsl.trs = gsl_multifit_nlinear_trs_lm);
            using jacobian_t = PolySingularJacobian<D>; // set to std::nullptr_t to compute numerically
            using hessian_t = PolySingularHessian<D>; // set to std::nullptr_t to compute numerically (also set fitparametersgsl.trs = gsl_multifit_nlinear_trs_lm);
            using transform_t = PolySingularTransform<I,D,M>;
            using regularization_t = PolySingularRegularization<D>;
        };
        
        struct PolySingular
        {
            template<typename I, typename D, U M> using type = PolySingularImpl<I, D, M>;
        };
        
    };
};

#endif
