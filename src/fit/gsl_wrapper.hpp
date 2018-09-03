#ifndef QMC_GSL_WRAPPER_H
#define QMC_GSL_WRAPPER_H

#include <algorithm> // std::max
#include <cassert> // assert
#include <cmath> // std::nan
#include <string> // std::to_string
#include <iostream> // std::endl
#include <iomanip> //  std::setw, std::setfill
#include <vector> // std::vector

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#include "../types/fit.hpp"

namespace integrators
{
    namespace fit
    {
        template<typename D, typename F1, typename F2>
        struct least_squares_wrapper_t {
            const F1 fit_function;
            const F2 fit_function_jacobian;
            const std::vector<D> x;
            const std::vector<D> y;
        };

        template<typename D, typename F1, typename F2>
        int fit_function_wrapper(const gsl_vector * parameters_vector, void *xyfunc_ptr, gsl_vector * f)
        {
            // Unpack data
            const std::vector<D>& x = reinterpret_cast<least_squares_wrapper_t<D,F1,F2>*>(xyfunc_ptr)->x;
            const std::vector<D>& y = reinterpret_cast<least_squares_wrapper_t<D,F1,F2>*>(xyfunc_ptr)->y;
            const F1& fit_function = reinterpret_cast<least_squares_wrapper_t<D,F1,F2>*>(xyfunc_ptr)->fit_function;

            // Compute deviates of fit function for input points
            for (size_t i = 0; i < x.size(); i++)
            {
                gsl_vector_set(f, i, static_cast<double>(fit_function(x[i], gsl_vector_const_ptr(parameters_vector,0)) - y[i]) );
            }

            return GSL_SUCCESS;
        }

        template<typename D, typename F1, typename F2>
        int fit_function_jacobian_wrapper(const gsl_vector * parameters_vector, void *xyfunc_ptr, gsl_matrix * J)
        {
            // Unpack data
            const std::vector<D>& x = reinterpret_cast<least_squares_wrapper_t<D,F1,F2>*>(xyfunc_ptr)->x;
            const F2& fit_function_jacobian = reinterpret_cast<least_squares_wrapper_t<D,F1,F2>*>(xyfunc_ptr)->fit_function_jacobian;

            for (size_t i = 0; i < x.size(); i++)
                for (size_t j = 0; j < fit_function_jacobian.num_parameters; j++)
                    gsl_matrix_set(J, i, j, static_cast<double>(fit_function_jacobian(x[i], gsl_vector_const_ptr(parameters_vector,0), j)) );

            return GSL_SUCCESS;
        }

        template<typename U>
        struct callback_params_t {
            const U& verbosity;
            Logger& logger;
        };

        template<typename U>
        void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w)
        {
            const U& verbosity = reinterpret_cast<callback_params_t<U>*>(params)->verbosity;
            Logger& logger = reinterpret_cast<callback_params_t<U>*>(params)->logger;

            gsl_vector *f = gsl_multifit_nlinear_residual(w);
            gsl_vector *x = gsl_multifit_nlinear_position(w);
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
            logger << std::left << std::setw(9) << std::setfill(separator) << "|f(x)| = ";
            logger << std::left << std::setw(num_width) << std::setfill(separator) << gsl_blas_dnrm2(f);
            logger << std::endl;
            logger.display_timing = display_timing;
        }

        template <typename D, typename U, typename F1, typename F2>
        std::vector<D> least_squares(F1& fit_function, F2& fit_function_jacobian, const std::vector<D>& x, const std::vector<D>& y, const U& verbosity, Logger& logger)
        {
            // fit parameters
            // TODO - Member variables or otherwise settable?
            const int maxiter = 40;
            const double xtol = 1e-10;
            const double gtol = 0.0;
            const double ftol = 0.0;

            const size_t num_points = x.size();
            const size_t num_parameters = fit_function.num_parameters;

            assert(x.size() == y.size());
            assert(num_points > num_parameters + 1);

            least_squares_wrapper_t<D,F1,F2> data = { fit_function, fit_function_jacobian, x, y };

            const gsl_multifit_nlinear_type *method = gsl_multifit_nlinear_trust;
            gsl_multifit_nlinear_workspace *w;
            gsl_multifit_nlinear_fdf fdf;
            gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters(); // TODO - set these?
            gsl_vector *f;
            gsl_matrix *J;
            gsl_matrix *covar = gsl_matrix_alloc(num_parameters, num_parameters);

            // define the function to be minimized
            fdf.f = fit_function_wrapper<D,F1,F2>;
            fdf.df = fit_function_jacobian_wrapper<D,F1,F2>;
            fdf.fvv = nullptr; // not using geodesic acceleration
            fdf.n = num_points;
            fdf.p = num_parameters;
            fdf.params = &data;

            double chisq,chisq0,chisq1,chisq2;
            int status, info;

            // allocate workspace with parameters
            w = gsl_multifit_nlinear_alloc(method, &fdf_params, num_points, num_parameters);

            std::vector<std::vector<D>> fit_parameters;

            for (int i = 1; i < 3; i++)
            {
                if (verbosity > 0)
                    logger << "-- running fit (run " << i << ")" << " --" << std::endl;

                static std::vector<double> initial_parameters;
                if (i == 1)
                    initial_parameters = { 1.1,1.0,0.0,0.0};
                else
                    initial_parameters = {-0.1,1.0,0.0,0.0};

                gsl_vector_view pv = gsl_vector_view_array(initial_parameters.data(), num_parameters);

                // initialize solver with starting point
                gsl_multifit_nlinear_init(&pv.vector, &fdf, w);

                // compute initial cost function
                f = gsl_multifit_nlinear_residual(w);
                gsl_blas_ddot(f, f, &chisq0);

                // solve the system with a maximum of "maxiter" iterations
                callback_params_t<U> callback_params{verbosity,logger};
                if (verbosity > 2)
                    status = gsl_multifit_nlinear_driver(maxiter, xtol, gtol, ftol, callback<U>, &callback_params, &info, w);
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
                for (size_t i = 0; i < num_parameters; i++)
                    this_fit_parameters.push_back( gsl_vector_get(w->x, i) );
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
                    for (size_t i = 0; i < num_parameters; i++)
                        logger << "fit_parameters[" << i << "] = " << this_fit_parameters[i] << " +/- " << c*sqrt(gsl_matrix_get(covar,i,i)*chisq/dof) << std::endl;
                    logger << "status = "              << gsl_strerror(status) << std::endl;
                    logger << "-----------" << std::endl;
                }

                if (i == 1)
                    chisq1 = chisq;
                else
                    chisq2 = chisq;
            }

            gsl_multifit_nlinear_free(w);
            gsl_matrix_free(covar);

            const size_t solution = chisq1 < chisq2 ? 0 : 1;

            if (verbosity > 1)
            {
                logger << "choosing solution " << solution+1 << std::endl;
                logger << "-----------" << std::endl;
            }

            return fit_parameters.at(solution);
        }
    };
};

#endif
