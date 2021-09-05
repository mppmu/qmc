#ifndef QMC_CORE_REDUCE_H
#define QMC_CORE_REDUCE_H

#include "../types/result.hpp"

namespace integrators
{
    namespace core
    {
        template <typename T>
        integrators::result<T> reduce(const std::vector<T>& r, const U n, const U m, std::vector<result<T>> & previous_iterations, const U& verbosity, const Logger& logger)
        {
            if (verbosity > 1)
            {
                logger << "-- qmc::reduce called --" << std::endl;
                for(const auto& previous_result : previous_iterations)
                {
                    logger << "previous_result: integral " << previous_result.integral << ", error " << previous_result.error << ", n " << previous_result.n << ", m " << previous_result.m << std::endl;
                }
            }

            T mean = {0.};
            T variance = {0.};
            U previous_m = 0;
            U previous_num_iterations = 0;
            U previous_num_evaluations = 0;
            if(!previous_iterations.empty())
            {
                result<T> & previous_res = previous_iterations.back();
                previous_num_iterations = previous_res.iterations;
                previous_num_evaluations = previous_res.evaluations;
                if(previous_res.n == n)
                {
                    if (verbosity>2) logger << "using additional shifts to improve previous iteration" << std::endl;
                    previous_m = previous_res.m;
                    mean = previous_res.integral*static_cast<T>(n);
                    variance = integrators::overloads::compute_variance_from_error(previous_res.error);
                    variance *= static_cast<T>(previous_res.m-1) * static_cast<T>(previous_res.m) * static_cast<T>(previous_res.n) * static_cast<T>(previous_res.n);
                    previous_iterations.pop_back();
                }
            }
            U r_size = r.size();
            for(U k = 0; k < m; k++)
            {
                T sum = {0.};
                T delta = {0.};
                for (U i = 0; i<r_size/m; i++)
                {
                    sum += r.at(k*r_size/m+i);
                }
                if (verbosity > 1) logger << "shift " << k+previous_m << " result: " << sum/static_cast<T>(n) << std::endl;
                // Compute Variance using online algorithm (Knuth, The Art of Computer Programming)
                delta = sum - mean;
                mean = mean + delta/(static_cast<T>(k+previous_m+1));
                variance = integrators::overloads::compute_variance(mean, variance, sum, delta);
            }
            T integral = mean/(static_cast<T>(n));
            variance = variance/( static_cast<T>(m+previous_m-1) * static_cast<T>(m+previous_m) * static_cast<T>(n) * static_cast<T>(n) ); // variance of the mean
            T error = integrators::overloads::compute_error(variance);
            previous_iterations.push_back({integral, error, n, m+previous_m, 1+previous_num_iterations, n*m+previous_num_evaluations});
            if (verbosity > 0)
                logger << "integral " << integral << ", error " << error << ", n " << n << ", m " << m+previous_m << std::endl;
            return {integral, error, n, m+previous_m, 1+previous_num_iterations, n*m+previous_num_evaluations};
        };
    };
};

#endif
