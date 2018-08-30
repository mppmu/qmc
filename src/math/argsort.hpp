#ifndef QMC_ARGSORT_H
#define QMC_ARGSORT_H

#include <algorithm> // std::sort
#include <numeric> // std::iota
#include <vector> // std::vector

namespace integrators
{
    namespace math
    {
        // argsort as suggested in https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
        template <typename T>
        std::vector<size_t> argsort(const std::vector<T> &v) {

          // initialize original index locations
          std::vector<size_t> idx(v.size());
          std::iota(idx.begin(), idx.end(), 0);

          // sort indexes based on comparing values in v
          std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

          // return vector of indices
          return idx;
        };
    };
};

#endif
