#ifndef QMC_SAMPLES_H
#define QMC_SAMPLES_H

#include <vector> // vector
#include "../math/mul_mod.hpp"

namespace integrators
{
    template <typename T, typename D, typename U = unsigned long long int>
    struct samples
    {
        std::vector<U> z;
        std::vector<D> d;
        std::vector<T> r;
        U n;

        D get_x(const U sample_index, const U dimension)
        {
            D mynull;
            return modf( integrators::math::mul_mod<D,D,U>(sample_index,z.at(dimension),n)/(static_cast<D>(n)) + d.at(dimension), &mynull);
        }
    };
};

#endif
