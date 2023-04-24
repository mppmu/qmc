#ifndef QMC_GENERATINGVECTORS_NONE
#define QMC_GENERATINGVECTORS_NONE

#include <vector>
#include <map>

namespace integrators
{
    namespace generatingvectors
    {
        inline std::map<U,std::vector<U>> none()
        {
            // for the use with the median qmc construction
            return std::map<U,std::vector<U>>();
        }
    };
};
#endif
