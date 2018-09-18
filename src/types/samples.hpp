#ifndef QMC_SAMPLES_H
#define QMC_SAMPLES_H

#include <vector> // vector
#include "../math/mul_mod.hpp"

namespace integrators
{
    template <typename T, typename D>
    struct samples
    {
        std::vector<U> z;
        std::vector<D> d;
        std::vector<T> r;
        U n;

        D get_x(const U sample_index, const U integration_variable_index)
        {
            D mynull;
            return modf( integrators::math::mul_mod<D,D>(sample_index,z.at(integration_variable_index),n)/(static_cast<D>(n)) + d.at(integration_variable_index), &mynull);
        }
    };
};

#endif
