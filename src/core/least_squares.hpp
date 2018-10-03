#ifndef QMC_LEAST_SQUARES_H
#define QMC_LEAST_SQUARES_H

#include <algorithm> // std::max
#include <cassert> // assert
#include <cmath> // std::nan
#include <cstddef> // std::nullptr_t
#include <sstream> // std::ostringstream
#include <string> // std::to_string
#include <iostream> // std::endl
#include <iomanip> //  std::setw, std::setfill
#include <vector> // std::vector
#include <iterator> // std::distance

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

namespace integrators
{
    namespace core
    {
        using fit_function_jacobian_wrapper_ptr = int (*)(const gsl_vector * , void *, gsl_matrix *);
        using fit_function_hessian_wrapper_ptr = int (*)(const gsl_vector *, const gsl_vector *, void *, gsl_vector *);

        template<typename D, typename F1, typename F2, typename F3>
        struct least_squares_wrapper_t {
            const F1 fit_function;
            const F2 fit_function_jacobian;
            const F3 fit_function_hessian;
            const std::vector<D> x;
            const std::vector<D> y;
        };

        template<typename D, typename F1, typename F2, typename F3>
        int fit_function_wrapper(const gsl_vector * parameters_vector, void *xyfunc_ptr, gsl_vector * f)
        {
            // Unpack data
            const std::vector<D>& x = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->x;
            const std::vector<D>& y = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->y;
            const F1& fit_function = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->fit_function;

            // Compute deviates of fit function for input points
            for (size_t i = 0; i < x.size(); i++)
            {
                gsl_vector_set(f, i, static_cast<double>(fit_function(x[i], gsl_vector_const_ptr(parameters_vector,0)) - y[i]) );
            }

            return GSL_SUCCESS;
        }

        template<typename D, typename F1, typename F2, typename F3>
        int fit_function_jacobian_wrapper(const gsl_vector * parameters_vector, void *xyfunc_ptr, gsl_matrix * J)
        {
            // Unpack data
            const std::vector<D>& x = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->x;
            const F2& fit_function_jacobian = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->fit_function_jacobian;

            for (size_t i = 0; i < x.size(); i++)
                for (size_t j = 0; j < fit_function_jacobian.num_parameters; j++)
                    gsl_matrix_set(J, i, j, static_cast<double>(fit_function_jacobian(x[i], gsl_vector_const_ptr(parameters_vector,0), j)) );

            return GSL_SUCCESS;
        }

        template<typename D, typename F1, typename F2, typename F3>
        int fit_function_hessian_wrapper(const gsl_vector * parameters_vector, const gsl_vector * v, void *xyfunc_ptr, gsl_vector * fvv)
        {
            // Unpack data
            const std::vector<D>& x = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->x;
            const F3& fit_function_hessian = reinterpret_cast<least_squares_wrapper_t<D,F1,F2,F3>*>(xyfunc_ptr)->fit_function_hessian;

            // Compute hessian of fit function
            for (size_t i = 0; i < x.size(); i++)
            {
                gsl_vector_set(fvv, i, static_cast<double>(fit_function_hessian(x[i], gsl_vector_const_ptr(v,0), gsl_vector_const_ptr(parameters_vector,0))) );
            }
            return GSL_SUCCESS;
        }

        struct callback_params_t {
            const U& verbosity;
            Logger& logger;
        };

        inline void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w)
        {
//            const U& verbosity = reinterpret_cast<callback_params_t*>(params)->verbosity;
            Logger& logger = reinterpret_cast<callback_params_t*>(params)->logger;

            gsl_vector *f = gsl_multifit_nlinear_residual(w);
            gsl_vector *x = gsl_multifit_nlinear_position(w);
            double avratio = gsl_multifit_nlinear_avratio(w);
            double rcond = std::nan("");

            // compute reciprocal condition number of J(x)
            if ( iter > 0 )
                gsl_multifit_nlinear_rcond(&rcond, w);

            const char separator       = ' ';
            const int name_width       = 11;
            const int small_name_width = 9;
            const int num_width        = 15;
            logger << std::left << std::setw(name_width) << std::setfill(separator) << "iter " + std::to_string(iter) + ": ";
            bool display_timing = logger.display_timing;
            logger.display_timing = false;
            for (size_t i = 0; i < x->size; i++)
            {
                logger << std::left << std::setw(small_name_width) << std::setfill(separator) << "p[" + std::to_string(i) + "] = ";
                logger << std::left << std::setw(num_width) << std::setfill(separator) << gsl_vector_get(x, i);
            }
            logger << std::left << std::setw(10) << std::setfill(separator) << "cond(J) = ";
            logger << std::left << std::setw(num_width) << std::setfill(separator) << 1.0 / rcond;
            logger << std::left << std::setw(10) << std::setfill(separator) << "|a|/|v| = ";
            logger << std::left << std::setw(num_width) << std::setfill(separator) << avratio;
            logger << std::left << std::setw(9) << std::setfill(separator) << "|f(x)| = ";
            logger << std::left << std::setw(num_width) << std::setfill(separator) << gsl_blas_dnrm2(f);
            logger << std::endl;
            logger.display_timing = display_timing;
        }

        template <typename D, typename F1, typename F2, typename F3>
        fit_function_jacobian_wrapper_ptr get_fit_function_jacobian_wrapper(const F2& fit_function_jacobian, const U& verbosity, Logger& logger)
        {
            if (verbosity > 1)
                logger << "using analytic jacobian" << std::endl;
            return fit_function_jacobian_wrapper<D,F1,F2,F3>;
        }
        template <typename D, typename F1, typename F2, typename F3>
        std::nullptr_t get_fit_function_jacobian_wrapper(const std::nullptr_t& fit_function_jacobian, const U& verbosity, Logger& logger)
        {
            if (verbosity > 1)
                logger << "using numeric jacobian" << std::endl;
            return nullptr;
        }

        template <typename D, typename F1, typename F2, typename F3>
        fit_function_hessian_wrapper_ptr get_fit_function_hessian_wrapper(const F3& fit_function_hessian, const U& verbosity, Logger& logger)
        {
            if (verbosity > 1)
                logger << "using analytic hessian" << std::endl;
            return fit_function_hessian_wrapper<D,F1,F2,F3>;
        }
        template <typename D, typename F1, typename F2, typename F3>
        std::nullptr_t get_fit_function_hessian_wrapper(const std::nullptr_t& fit_function_hessian, const U& verbosity, Logger& logger)
        {
            if (verbosity > 1)
                logger << "using numeric hessian" << std::endl;
            return nullptr;
        }

        template <typename D, typename F1, typename F2, typename F3>
        std::vector<D> least_squares(F1& fit_function, F2& fit_function_jacobian, F3& fit_function_hessian, const std::vector<D>& x, const std::vector<D>& y, const U& verbosity, Logger& logger, const size_t maxiter, const double xtol, const double gtol, const double ftol, gsl_multifit_nlinear_parameters fitparametersgsl)
        {
            const size_t num_points = x.size();
            const size_t num_parameters = fit_function.num_parameters;

            assert(x.size() == y.size());
            assert(num_points > num_parameters + 1);

            least_squares_wrapper_t<D,F1,F2,F3> data = { fit_function, fit_function_jacobian, fit_function_hessian, x, y };

            const gsl_multifit_nlinear_type *method = gsl_multifit_nlinear_trust;
            gsl_multifit_nlinear_workspace *w;
            gsl_multifit_nlinear_fdf fdf;
            gsl_multifit_nlinear_parameters fdf_params = fitparametersgsl;
            gsl_vector *f;
            gsl_matrix *J;
            gsl_matrix *covar = gsl_matrix_alloc(num_parameters, num_parameters);

            // define the function to be minimized
            fdf.f = fit_function_wrapper<D,F1,F2,F3>;
            fdf.df = get_fit_function_jacobian_wrapper<D,F1,F2,F3>(fit_function_jacobian, verbosity, logger);
            fdf.fvv = get_fit_function_hessian_wrapper<D,F1,F2,F3>(fit_function_hessian, verbosity, logger);
            fdf.n = num_points;
            fdf.p = num_parameters;
            fdf.params = &data;
            
            // compute dx/dy of input points, which should be used as an additional weight in the evaluation of chisq
            std::vector<D> dxdy(x.size());
            D maxwgt = 0.;

            const size_t nsteps = 1; 
            for (size_t i = 0; i < x.size(); i++)
            {
                D dy = (i<nsteps) ? D(0) : -y[i-nsteps];  
                D dx = (i<nsteps) ? D(0) : -x[i-nsteps];
                if(i != x.size()-nsteps)
                {
                    dy += y[i+nsteps];
                    dx += x[i+nsteps];
                }
                else
                {
                    dy += D(1);
                    dx += D(1);
                }
                dxdy[i] = dx/dy;
                
                maxwgt=std::max(maxwgt,dxdy[i]);
            }
            
            // the gsl fit doesn't seem to work with weights>1 
            for(size_t i=0; i< x.size(); i++)
            {
                dxdy[i]/=maxwgt;
            }
            
            gsl_vector_view wgt = gsl_vector_view_array(dxdy.data(), dxdy.size());

            double chisq,chisq0;
            int status, info;

            // allocate workspace with parameters
            w = gsl_multifit_nlinear_alloc(method, &fdf_params, num_points, num_parameters);

            std::vector<std::vector<D>> fit_parameters;
            std::vector<double> fit_chisqs;
            fit_chisqs.reserve(fit_function.initial_parameters.size());
            for (size_t i = 0; i < fit_function.initial_parameters.size(); i++)
            {
                std::vector<double> initial_parameters(fit_function.initial_parameters.at(i).begin(),fit_function.initial_parameters.at(i).end()); // note: cast to double

                if( initial_parameters.size() != fit_function.num_parameters)
                    throw std::domain_error("least_squares called with incorrect number of initial_parameters (" + std::to_string(initial_parameters.size()) + "), expected " +  std::to_string(fit_function.num_parameters) + " parameters");

                if (verbosity > 0)
                {
                    logger << "-- running fit (run " << i << ")" << " --" << std::endl;
                    std::ostringstream initial_parameters_stream;
                    for(const auto& elem: initial_parameters)
                        initial_parameters_stream << elem << " ";
                    logger << "with initial_parameters " << initial_parameters_stream.str() << std::endl;
                }

                gsl_vector_view pv = gsl_vector_view_array(initial_parameters.data(), num_parameters);

                // initialize solver with starting point
                gsl_multifit_nlinear_winit(&pv.vector, &wgt.vector, &fdf, w);

                // compute initial cost function
                f = gsl_multifit_nlinear_residual(w);
                gsl_blas_ddot(f, f, &chisq0);

                // solve the system with a maximum of "maxiter" iterations
                callback_params_t callback_params{verbosity,logger};
                if (verbosity > 2)
                    status = gsl_multifit_nlinear_driver(maxiter, xtol, gtol, ftol, callback, &callback_params, &info, w);
                else
                    status = gsl_multifit_nlinear_driver(maxiter, xtol, gtol, ftol, nullptr, nullptr, &info, w);

                // compute covariance of best fit parameters
                J = gsl_multifit_nlinear_jac(w);
                gsl_multifit_nlinear_covar(J, 0., covar);

                // compute final cost
                gsl_blas_ddot(f, f, &chisq);

                // store fit parameters
                std::vector<D> this_fit_parameters;
                this_fit_parameters.reserve(num_parameters);
                for (size_t j = 0; j < num_parameters; j++)
                    this_fit_parameters.push_back( gsl_vector_get(w->x, j) );
                fit_parameters.push_back( this_fit_parameters );

                // Report output of fit function
                if (verbosity > 1)
                {
                    double dof = num_points - num_parameters - 1;
                    double c = std::max(1., sqrt(chisq/dof));

                    logger << "-- fit output (run " << i << ")" << " --" << std::endl;
                    logger << "summary from method "   << gsl_multifit_nlinear_name(w) << " " << gsl_multifit_nlinear_trs_name(w) << std::endl;
                    logger << "number of iterations: " << gsl_multifit_nlinear_niter(w) << std::endl;
                    logger << "function evaluations: " << fdf.nevalf << std::endl;
                    logger << "Jacobian evaluations: " << fdf.nevaldf << std::endl;
                    logger << "Hessian evaluations: " << fdf.nevalfvv << std::endl;
                    if (info == 0)
                        logger << "reason for stopping: " << "maximal number of iterations (maxiter)" << std::endl;
                    else if (info == 1)
                        logger << "reason for stopping: " << "small step size (xtol)" << std::endl;
                    else if (info == 2)
                        logger << "reason for stopping: " << "small gradient (gtol)" << std::endl;
                    else
                        logger << "reason for stopping: " << "unknown" << std::endl;
                    logger << "initial |f(x)| = "      << sqrt(chisq0) << std::endl;;
                    logger << "final   |f(x)| = "      << sqrt(chisq) << std::endl;
                    logger << "chisq/dof = "           << chisq/dof << std::endl;
                    for (size_t j = 0; j < num_parameters; j++)
                        logger << "fit_parameters[" << j << "] = " << this_fit_parameters.at(j) << " +/- " << c*sqrt(gsl_matrix_get(covar,j,j)*chisq/dof) << std::endl;
                    logger << "status = "              << gsl_strerror(status) << std::endl;
                    logger << "-----------" << std::endl;
                }

                fit_chisqs.push_back(chisq);

            }


            gsl_multifit_nlinear_free(w);
            gsl_matrix_free(covar);

            // get index of best fit (minimum chisq)
            const long best_fit_index = std::distance(fit_chisqs.begin(), std::min_element(fit_chisqs.begin(),fit_chisqs.end()));

            if (verbosity > 0)
            {
                if (verbosity>2) logger << "choosing fit run " << best_fit_index << std::endl;
                std::ostringstream final_parameters_stream;
                for(const auto& elem: fit_parameters.at(best_fit_index))
                    final_parameters_stream << elem << " ";
                logger << "fit final_parameters " << final_parameters_stream.str() << std::endl;
                logger << "-----------" << std::endl;
            }

            return fit_parameters.at(best_fit_index);
        }
    };
};

#endif
