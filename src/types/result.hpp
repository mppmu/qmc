#ifndef QMC_RESULT_H
#define QMC_RESULT_H

namespace integrators
{
    template <typename T>
    struct result
    {
        T integral;
        T error;
        U n;
        U m;
        U iterations;
        U evaluations;
    };
};

#endif
