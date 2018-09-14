#ifndef QMC_GENERATINGVECTORS_CBCPT_CFFTW1_6_H
#define QMC_GENERATINGVECTORS_CBCPT_CFFTW1_6_H

#include <vector>
#include <map>

namespace integrators
{
    namespace generatingvectors
    {
        inline std::map<U,std::vector<U>> cbcpt_cfftw1_6()
        {

            // Vectors generated using custom CBC tool based on FFTW
            // Settings:
            // s = 100
            // omega=inline('2*pi^2*(x.^2-x+1/6)')
            // gamma=1/s
            // beta=1

            std::map<U,std::vector<U>> generatingvectors;

            generatingvectors[2500000001]={1,1056092002,604902782,1140518443,1168484358,678540231};
            generatingvectors[3010560001]={1,1265039176,710224583,570392900,246175051,776375237};
            generatingvectors[3527193601]={1,1477280710,1631679535,500900763,1337951012,990240443};
            generatingvectors[4046192641]={1,1545350222,1821675932,852726071,257150351,1540501786};
            generatingvectors[4515840001]={1,1895743314,1978110099,1051107732,1249084094,95135867};
            generatingvectors[5040947521]={1,2084035980,585653597,448523180,856444223,2389197079};
            generatingvectors[5505024001]={1,2280282288,503769990,2547746687,2668753240,2100976149};
            generatingvectors[6165626881]={1,2360163115,1727923807,3043833953,2316665784,2702804871};
            generatingvectors[6561000001]={1,1812543072,1410669934,1037177071,1156985284,3184493703};
            generatingvectors[7112448001]={1,2716396872,3010889702,2894020956,1485836748,1959799747};
            generatingvectors[7501410001]={1,2757177450,879378109,2460097563,3069036981,3181359993};
            generatingvectors[10088401751]={1,4180383535,3234338499,4352977744,1736039557,4095101376};
            generatingvectors[15173222401]={1,5795573206,4481927636,1112677318,2916664894,2804891062};

            return generatingvectors;

        }
    };
};
#endif
