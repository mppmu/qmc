#ifndef QMC_GENERATINGVECTORS_CBCPT_DN2_6_H
#define QMC_GENERATINGVECTORS_CBCPT_DN2_6_H

#include <vector>
#include <map>

namespace integrators
{
    namespace generatingvectors
    {
        template <typename U>
        std::map<U,std::vector<U>> cbcpt_dn2_6()
        {

            // Vectors generated using Dirk Nuyens' fastrank1pt.m tool https://people.cs.kuleuven.be/~dirk.nuyens/fast-cbc
            // Settings:
            // s = 100
            // omega=inline('2*pi^2*(x.^2-x+1/6)')
            // gamma=1/s
            // beta=1

            // Used for integration in arXiv:1608.04798, arXiv:1604.06447, arXiv:1802.00349

            std::map<U,std::vector<U>> generatingvectors;

            generatingvectors[65521] = {1,18303,27193,16899,31463,13841};
            generatingvectors[131071] = {1,49763,21432,15971,52704,48065};
            generatingvectors[196597] = {1,72610,13914,40202,16516,29544};
            generatingvectors[262139] = {1,76811,28708,119567,126364,5581};
            generatingvectors[327673] = {1,125075,70759,81229,99364,145331};
            generatingvectors[393209] = {1,150061,176857,160143,163763,27779};
            generatingvectors[458747] = {1,169705,198529,128346,134850,173318};
            generatingvectors[524287] = {1,153309,134071,36812,159642,245846};
            generatingvectors[655357] = {1,253462,69526,294762,304980,238532};
            generatingvectors[786431] = {1,300187,232015,63830,343869,39791};
            generatingvectors[982981] = {1,360960,73766,194632,51680,293702};
            generatingvectors[1245169] = {1,368213,239319,593224,147860,546740};
            generatingvectors[1572853] = {1,459925,736430,70288,373919,634109};
            generatingvectors[1966079] = {1,826127,686058,926897,417836,183049};
            generatingvectors[2359267] = {1,696379,1060519,640757,812754,262923};
            generatingvectors[2949119] = {1,1090495,479029,595914,64689,895947};
            generatingvectors[3670013] = {1,1357095,1026979,857015,644825,1129417};
            generatingvectors[4587503] = {1,1742417,1399874,672080,1827715,1488353};
            generatingvectors[5767129] = {1,2210064,514295,1675989,137965,1611055};
            generatingvectors[7208951] = {1,3144587,1709091,872489,489266,2288306};
            generatingvectors[8933471] = {1,3453775,2110983,261972,2555740,1086124};
            generatingvectors[12506869] = {1,3663001,2298621,853317,2983823,5576578};
            generatingvectors[17509627] = {1,6426637,1486159,2528828,866597,1015123};
            generatingvectors[24513479] = {1,9363157,10935868,7904120,7202893,10833044};
            generatingvectors[34318871] = {1,14408021,2474791,14056163,13619371,8142161};
            generatingvectors[48046423] = {1,17766606,14390535,18752150,15489536,22204578};
            generatingvectors[67264993] = {1,26061396,18907982,30760850,28273663,360289};
            generatingvectors[94170997] = {1,34493943,45822183,33604771,17761662,27235450};
            generatingvectors[131839397] = {1,50467100,12217927,32766578,62069641,43610269};
            generatingvectors[184575163] = {1,70104887,26696463,66178896,33835785,44887749};
            generatingvectors[258405233] = {1,93229880,121240317,81359405,13132851,3566987};
            generatingvectors[361767331] = {1,151870194,126971921,92157910,95131599,23325957};
            generatingvectors[506474291] = {1,187358266,98361527,30873241,97613632,101655550};
            generatingvectors[658416589] = {1,241782268,273651328,292254504,308395171,128636084};
            generatingvectors[855941587] = {1,330894152,231467230,413890922,328644676,277687765};
            generatingvectors[1112724071] = {1,469999985,500083067,198639601,547255654,47506555};
            generatingvectors[1446541331] = {1,597774846,391859496,244464942,540060630,431672305};
            generatingvectors[1735849631] = {1,728493464,221397948,296306937,119145578,528265440};
            generatingvectors[2083019591] = {1,863176824,162064764,556032311,529803849,822974245};
            generatingvectors[2499623531] = {1,946563710,446148663,365195622,977705092,1214380109};

            return generatingvectors;

        }
    };
};
#endif
