/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 100_hh_sector_demo.cpp -o 100_hh_sector_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 100_hh_sector_demo.cpp -o 100_hh_sector_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <iomanip>

#include "qmc.hpp"

#ifdef __CUDACC__
#include <thrust/complex.h>
typedef thrust::complex<double> dcmplx;
#else
#include <complex>
typedef std::complex<double> dcmplx;
#endif

struct ReduzeF1L2_021111010ord0f3_t {
    const unsigned long long int number_of_integration_variables = 5;
#ifdef __CUDACC__
    __host__ __device__
#endif
dcmplx operator()(const double xxx[]) const
    {

    // Run 4068 (bad)
//    double p[11];
//    p[0] = 4;
//    p[1] = -2.5221;
//    p[2] = 0.99255;
//    p[3] = 0.51818;
//    p[4] = 1;
//    p[5] = 0.178875;
//    p[6] = 0.264935;
//    p[7] = 0.512265;
//    p[8] = 0.198156;
//    p[9] = 0.238695;
//    p[10] = 1;

    // Run 4160 (vbad)
    double p[11];
    p[0] = 4;
    p[1] = -2.81415;
    p[2] = 0.994472;
    p[3] = 0.519183;
    p[4] = 1;
    p[5] = 0.186205;
    p[6] = 0.25513;
    p[7] = 0.473155;
    p[8] = 0.200002;
    p[9] = 0.234535;
    p[10] = 1;

//    double p[11];
//    p[0] = 4;
//    p[1] = -1.81415;
//    p[2] = 1.994472;
//    p[3] = 0.219183;
//    p[4] = 1;
//    p[5] = 0.186205;
//    p[6] = 0.35513;
//    p[7] = 0.273155;
//    p[8] = 0.100002;
//    p[9] = 0.534535;
//    p[10] = 1;

double x[5];

x[0] = xxx[3];
x[1] = xxx[1];
x[2] = xxx[2];
x[3] = xxx[0];
x[4] = xxx[4];

double esx[2];
double em[2];
double lrs[5];
double x0=x[0];
double x1=x[1];
double x2=x[2];
double x3=x[3];
double x4=x[4];
esx[0]=p[0];
esx[1]=p[1];
em[0]=p[2];
em[1]=p[3];
double lambda=p[4];
lrs[0]=p[5];
lrs[1]=p[6];
lrs[2]=p[7];
lrs[3]=p[8];
lrs[4]=p[9];
double bi=p[10];
dcmplx FOUT;
dcmplx MYI(0.,1.);
dcmplx y[351];
y[1]=em[0];
y[2]=em[1];
y[3]=2.*x0*y[1];
y[4]=2.*x1*y[1];
y[5]=2.*y[1];
y[6]=2.*x0*x4*y[1];
y[7]=2.*x1*x4*y[1];
y[8]=2.*x2*x4*y[1];
y[9]=2.*x3*x4*y[1];
y[10]=-(x3*x4*y[2]);
y[11]=esx[0];
y[12]=x4*y[1];
y[13]=2.*x2*y[1];
y[14]=2.*x0*x2*y[1];
y[15]=2.*x1*x2*y[1];
y[16]=pow(x2,2);
y[17]=2.*x3*y[1];
y[18]=2.*x0*x3*y[1];
y[19]=2.*x1*x3*y[1];
y[20]=pow(x3,2);
y[21]=lrs[0];
y[22]=-1.+x1;
y[23]=y[1]*y[16];
y[24]=2.*x2*x3*y[1];
y[25]=y[1]*y[20];
y[26]=-(x3*y[2]);
y[27]=-(x2*x3*y[2]);
y[28]=-(x2*y[11]);
y[29]=y[1]+y[3]+y[4]+y[13]+y[14]+y[15]+y[17]+y[18]+y[19]+y[23]+y[24]+y[25]+y\
[26]+y[27]+y[28];
y[30]=pow(x0,2);
y[31]=esx[1];
y[32]=lrs[1];
y[33]=-1.+x2;
y[34]=2.*x4*y[1];
y[35]=-(x4*y[11]);
y[36]=y[5]+y[6]+y[7]+y[8]+y[9]+y[10]+y[34]+y[35];
y[37]=pow(lambda,2);
y[38]=-1.+x0;
y[39]=x2*x4*y[1];
y[40]=x3*x4*y[1];
y[41]=y[1]+y[12]+y[39]+y[40];
y[42]=-MYI;
y[43]=pow(x1,2);
y[44]=lrs[2];
y[45]=y[1]*y[30];
y[46]=2.*x0*x1*y[1];
y[47]=y[1]*y[43];
y[48]=-2.*lambda*y[1]*y[21];
y[49]=6.*lambda*y[1]*y[21]*y[30];
y[50]=-2.*lambda*x1*y[1]*y[21];
y[51]=4.*lambda*x0*x1*y[1]*y[21];
y[52]=-2.*lambda*x2*y[1]*y[21];
y[53]=4.*lambda*x0*x2*y[1]*y[21];
y[54]=-2.*lambda*x3*y[1]*y[21];
y[55]=4.*lambda*x0*x3*y[1]*y[21];
y[56]=-(lambda*x4*y[1]*y[21]);
y[57]=-2.*lambda*x0*x4*y[1]*y[21];
y[58]=6.*lambda*x4*y[1]*y[21]*y[30];
y[59]=-2.*lambda*x1*x4*y[1]*y[21];
y[60]=4.*lambda*x0*x1*x4*y[1]*y[21];
y[61]=-2.*lambda*x2*x4*y[1]*y[21];
y[62]=6.*lambda*x2*x4*y[1]*y[21]*y[30];
y[63]=-2.*lambda*x1*x2*x4*y[1]*y[21];
y[64]=4.*lambda*x0*x1*x2*x4*y[1]*y[21];
y[65]=-(lambda*x4*y[1]*y[16]*y[21]);
y[66]=2.*lambda*x0*x4*y[1]*y[16]*y[21];
y[67]=-2.*lambda*x3*x4*y[1]*y[21];
y[68]=6.*lambda*x3*x4*y[1]*y[21]*y[30];
y[69]=-2.*lambda*x1*x3*x4*y[1]*y[21];
y[70]=4.*lambda*x0*x1*x3*x4*y[1]*y[21];
y[71]=-2.*lambda*x2*x3*x4*y[1]*y[21];
y[72]=4.*lambda*x0*x2*x3*x4*y[1]*y[21];
y[73]=-(lambda*x4*y[1]*y[20]*y[21]);
y[74]=2.*lambda*x0*x4*y[1]*y[20]*y[21];
y[75]=lambda*x3*x4*y[2]*y[21];
y[76]=-2.*lambda*x0*x3*x4*y[2]*y[21];
y[77]=lambda*x2*x3*x4*y[2]*y[21];
y[78]=-2.*lambda*x0*x2*x3*x4*y[2]*y[21];
y[79]=lambda*x2*x4*y[11]*y[21];
y[80]=-2.*lambda*x0*x2*x4*y[11]*y[21];
y[81]=lambda*x3*y[21]*y[31];
y[82]=-2.*lambda*x0*x3*y[21]*y[31];
y[83]=y[42]+y[48]+y[49]+y[50]+y[51]+y[52]+y[53]+y[54]+y[55]+y[56]+y[57]+y[58\
]+y[59]+y[60]+y[61]+y[62]+y[63]+y[64]+y[65]+y[66]+y[67]+y[68]+y[69]+y[70]+y\
[71]+y[72]+y[73]+y[74]+y[75]+y[76]+y[77]+y[78]+y[79]+y[80]+y[81]+y[82];
y[84]=-2.*lambda*y[1]*y[32];
y[85]=-2.*lambda*x0*y[1]*y[32];
y[86]=4.*lambda*x0*x1*y[1]*y[32];
y[87]=6.*lambda*y[1]*y[32]*y[43];
y[88]=-2.*lambda*x2*y[1]*y[32];
y[89]=4.*lambda*x1*x2*y[1]*y[32];
y[90]=-2.*lambda*x3*y[1]*y[32];
y[91]=4.*lambda*x1*x3*y[1]*y[32];
y[92]=-(lambda*x4*y[1]*y[32]);
y[93]=-2.*lambda*x0*x4*y[1]*y[32];
y[94]=-2.*lambda*x1*x4*y[1]*y[32];
y[95]=4.*lambda*x0*x1*x4*y[1]*y[32];
y[96]=6.*lambda*x4*y[1]*y[32]*y[43];
y[97]=-2.*lambda*x2*x4*y[1]*y[32];
y[98]=-2.*lambda*x0*x2*x4*y[1]*y[32];
y[99]=4.*lambda*x0*x1*x2*x4*y[1]*y[32];
y[100]=6.*lambda*x2*x4*y[1]*y[32]*y[43];
y[101]=-(lambda*x4*y[1]*y[16]*y[32]);
y[102]=2.*lambda*x1*x4*y[1]*y[16]*y[32];
y[103]=-2.*lambda*x3*x4*y[1]*y[32];
y[104]=-2.*lambda*x0*x3*x4*y[1]*y[32];
y[105]=4.*lambda*x0*x1*x3*x4*y[1]*y[32];
y[106]=6.*lambda*x3*x4*y[1]*y[32]*y[43];
y[107]=-2.*lambda*x2*x3*x4*y[1]*y[32];
y[108]=4.*lambda*x1*x2*x3*x4*y[1]*y[32];
y[109]=-(lambda*x4*y[1]*y[20]*y[32]);
y[110]=2.*lambda*x1*x4*y[1]*y[20]*y[32];
y[111]=lambda*x3*y[2]*y[32];
y[112]=-2.*lambda*x1*x3*y[2]*y[32];
y[113]=lambda*x3*x4*y[2]*y[32];
y[114]=-2.*lambda*x1*x3*x4*y[2]*y[32];
y[115]=lambda*x2*x3*x4*y[2]*y[32];
y[116]=-2.*lambda*x1*x2*x3*x4*y[2]*y[32];
y[117]=lambda*y[11]*y[32];
y[118]=-2.*lambda*x1*y[11]*y[32];
y[119]=lambda*x2*x4*y[11]*y[32];
y[120]=-2.*lambda*x1*x2*x4*y[11]*y[32];
y[121]=y[42]+y[84]+y[85]+y[86]+y[87]+y[88]+y[89]+y[90]+y[91]+y[92]+y[93]+y[9\
4]+y[95]+y[96]+y[97]+y[98]+y[99]+y[100]+y[101]+y[102]+y[103]+y[104]+y[105]+\
y[106]+y[107]+y[108]+y[109]+y[110]+y[111]+y[112]+y[113]+y[114]+y[115]+y[116\
]+y[117]+y[118]+y[119]+y[120];
y[122]=-1.+x3;
y[123]=-y[2];
y[124]=-(x0*x3*y[2]);
y[125]=-(x1*x3*y[2]);
y[126]=-(x0*y[11]);
y[127]=-(x1*y[11]);
y[128]=y[3]+y[4]+y[14]+y[15]+y[18]+y[19]+y[45]+y[46]+y[47]+y[124]+y[125]+y[1\
26]+y[127];
y[129]=-2.*x0*x1*y[21]*y[22]*y[29]*y[32]*y[37]*y[38]*y[41];
y[130]=lambda*x1*y[22]*y[29]*y[32]*y[83];
y[131]=y[129]+y[130];
y[132]=lrs[3];
y[133]=-(x4*y[2]);
y[134]=-(x2*x4*y[2]);
y[135]=2.*x0*x1*y[21]*y[22]*y[29]*y[32]*y[37]*y[38]*y[41];
y[136]=-(lambda*x0*y[21]*y[29]*y[38]*y[121]);
y[137]=y[135]+y[136];
y[138]=-2.*lambda*y[1]*y[44];
y[139]=-2.*lambda*x0*y[1]*y[44];
y[140]=-2.*lambda*x1*y[1]*y[44];
y[141]=4.*lambda*x0*x2*y[1]*y[44];
y[142]=4.*lambda*x1*x2*y[1]*y[44];
y[143]=6.*lambda*y[1]*y[16]*y[44];
y[144]=-2.*lambda*x3*y[1]*y[44];
y[145]=4.*lambda*x2*x3*y[1]*y[44];
y[146]=-2.*lambda*x0*x4*y[1]*y[44];
y[147]=-(lambda*x4*y[1]*y[30]*y[44]);
y[148]=-2.*lambda*x1*x4*y[1]*y[44];
y[149]=-2.*lambda*x0*x1*x4*y[1]*y[44];
y[150]=-(lambda*x4*y[1]*y[43]*y[44]);
y[151]=2.*lambda*x2*x4*y[1]*y[30]*y[44];
y[152]=4.*lambda*x0*x1*x2*x4*y[1]*y[44];
y[153]=2.*lambda*x2*x4*y[1]*y[43]*y[44];
y[154]=6.*lambda*x0*x4*y[1]*y[16]*y[44];
y[155]=6.*lambda*x1*x4*y[1]*y[16]*y[44];
y[156]=-2.*lambda*x0*x3*x4*y[1]*y[44];
y[157]=-2.*lambda*x1*x3*x4*y[1]*y[44];
y[158]=4.*lambda*x0*x2*x3*x4*y[1]*y[44];
y[159]=4.*lambda*x1*x2*x3*x4*y[1]*y[44];
y[160]=lambda*x3*y[2]*y[44];
y[161]=-2.*lambda*x2*x3*y[2]*y[44];
y[162]=lambda*x0*x3*x4*y[2]*y[44];
y[163]=lambda*x1*x3*x4*y[2]*y[44];
y[164]=-2.*lambda*x0*x2*x3*x4*y[2]*y[44];
y[165]=-2.*lambda*x1*x2*x3*x4*y[2]*y[44];
y[166]=lambda*y[11]*y[44];
y[167]=-2.*lambda*x2*y[11]*y[44];
y[168]=lambda*x0*x4*y[11]*y[44];
y[169]=lambda*x1*x4*y[11]*y[44];
y[170]=-2.*lambda*x0*x2*x4*y[11]*y[44];
y[171]=-2.*lambda*x1*x2*x4*y[11]*y[44];
y[172]=y[42]+y[138]+y[139]+y[140]+y[141]+y[142]+y[143]+y[144]+y[145]+y[146]+\
y[147]+y[148]+y[149]+y[150]+y[151]+y[152]+y[153]+y[154]+y[155]+y[156]+y[157\
]+y[158]+y[159]+y[160]+y[161]+y[162]+y[163]+y[164]+y[165]+y[166]+y[167]+y[1\
68]+y[169]+y[170]+y[171];
y[173]=-(x0*y[2]);
y[174]=-(x1*y[2]);
y[175]=-(x0*x2*y[2]);
y[176]=-(x1*x2*y[2]);
y[177]=y[3]+y[4]+y[14]+y[15]+y[18]+y[19]+y[45]+y[46]+y[47]+y[173]+y[174]+y[1\
75]+y[176];
y[178]=-2.*x0*x1*y[21]*y[22]*y[32]*y[36]*y[37]*y[38]*y[41];
y[179]=lambda*x1*y[22]*y[32]*y[36]*y[83];
y[180]=y[178]+y[179];
y[181]=2.*x0*x1*y[21]*y[22]*y[32]*y[36]*y[37]*y[38]*y[41];
y[182]=-(lambda*x0*y[21]*y[36]*y[38]*y[121]);
y[183]=y[181]+y[182];
y[184]=pow(y[41],2);
y[185]=-4.*x0*x1*y[21]*y[22]*y[32]*y[37]*y[38]*y[184];
y[186]=y[83]*y[121];
y[187]=y[185]+y[186];
y[188]=-1.+x4;
y[189]=y[5]+y[6]+y[7]+y[8]+y[9]+y[34]+y[123]+y[133]+y[134];
y[190]=-y[31];
y[191]=y[5]+y[6]+y[7]+y[8]+y[9]+y[34]+y[133]+y[134]+y[190];
y[192]=-(x0*x4*y[2]);
y[193]=-(x1*x4*y[2]);
y[194]=y[5]+y[6]+y[7]+y[123]+y[192]+y[193];
y[195]=-(x0*x1*y[21]*y[22]*y[29]*y[32]*y[37]*y[38]*y[189]);
y[196]=x0*x1*y[21]*y[22]*y[29]*y[32]*y[37]*y[38]*y[191];
y[197]=y[195]+y[196];
y[198]=lambda*x2*y[33]*y[36]*y[44]*y[197];
y[199]=-2.*x0*x1*y[21]*y[22]*y[32]*y[37]*y[38]*y[41]*y[191];
y[200]=lambda*x1*y[22]*y[32]*y[83]*y[189];
y[201]=y[199]+y[200];
y[202]=2.*x0*x1*y[21]*y[22]*y[32]*y[37]*y[38]*y[41]*y[189];
y[203]=-(lambda*x0*y[21]*y[38]*y[121]*y[191]);
y[204]=y[202]+y[203];
y[205]=-(lambda*x2*y[33]*y[36]*y[44]*y[131]);
y[206]=lambda*x2*y[33]*y[36]*y[44]*y[137];
y[207]=lambda*x2*y[33]*y[44]*y[128]*y[187];
y[208]=y[205]+y[206]+y[207];
y[209]=lrs[4];
y[210]=lambda*x2*y[33]*y[44]*y[128]*y[201];
y[211]=-(lambda*x2*y[33]*y[44]*y[131]*y[194]);
y[212]=y[198]+y[210]+y[211];
y[213]=x0*x1*y[21]*y[22]*y[32]*y[36]*y[37]*y[38]*y[189];
y[214]=-(x0*x1*y[21]*y[22]*y[32]*y[36]*y[37]*y[38]*y[191]);
y[215]=y[213]+y[214];
y[216]=lambda*x2*y[33]*y[44]*y[128]*y[180];
y[217]=-(y[131]*y[172]);
y[218]=y[216]+y[217];
y[219]=-2.*lambda*y[1]*y[132];
y[220]=-2.*lambda*x0*y[1]*y[132];
y[221]=-2.*lambda*x1*y[1]*y[132];
y[222]=-2.*lambda*x2*y[1]*y[132];
y[223]=4.*lambda*x0*x3*y[1]*y[132];
y[224]=4.*lambda*x1*x3*y[1]*y[132];
y[225]=4.*lambda*x2*x3*y[1]*y[132];
y[226]=6.*lambda*y[1]*y[20]*y[132];
y[227]=-2.*lambda*x0*x4*y[1]*y[132];
y[228]=-(lambda*x4*y[1]*y[30]*y[132]);
y[229]=-2.*lambda*x1*x4*y[1]*y[132];
y[230]=-2.*lambda*x0*x1*x4*y[1]*y[132];
y[231]=-(lambda*x4*y[1]*y[43]*y[132]);
y[232]=-2.*lambda*x0*x2*x4*y[1]*y[132];
y[233]=-2.*lambda*x1*x2*x4*y[1]*y[132];
y[234]=2.*lambda*x3*x4*y[1]*y[30]*y[132];
y[235]=4.*lambda*x0*x1*x3*x4*y[1]*y[132];
y[236]=2.*lambda*x3*x4*y[1]*y[43]*y[132];
y[237]=4.*lambda*x0*x2*x3*x4*y[1]*y[132];
y[238]=4.*lambda*x1*x2*x3*x4*y[1]*y[132];
y[239]=6.*lambda*x0*x4*y[1]*y[20]*y[132];
y[240]=6.*lambda*x1*x4*y[1]*y[20]*y[132];
y[241]=lambda*y[2]*y[132];
y[242]=lambda*x1*y[2]*y[132];
y[243]=lambda*x2*y[2]*y[132];
y[244]=-2.*lambda*x3*y[2]*y[132];
y[245]=-2.*lambda*x1*x3*y[2]*y[132];
y[246]=-2.*lambda*x2*x3*y[2]*y[132];
y[247]=lambda*x0*x4*y[2]*y[132];
y[248]=lambda*x1*x4*y[2]*y[132];
y[249]=lambda*x0*x2*x4*y[2]*y[132];
y[250]=lambda*x1*x2*x4*y[2]*y[132];
y[251]=-2.*lambda*x0*x3*x4*y[2]*y[132];
y[252]=-2.*lambda*x1*x3*x4*y[2]*y[132];
y[253]=-2.*lambda*x0*x2*x3*x4*y[2]*y[132];
y[254]=-2.*lambda*x1*x2*x3*x4*y[2]*y[132];
y[255]=lambda*x0*y[31]*y[132];
y[256]=-2.*lambda*x0*x3*y[31]*y[132];
y[257]=y[42]+y[219]+y[220]+y[221]+y[222]+y[223]+y[224]+y[225]+y[226]+y[227]+\
y[228]+y[229]+y[230]+y[231]+y[232]+y[233]+y[234]+y[235]+y[236]+y[237]+y[238\
]+y[239]+y[240]+y[241]+y[242]+y[243]+y[244]+y[245]+y[246]+y[247]+y[248]+y[2\
49]+y[250]+y[251]+y[252]+y[253]+y[254]+y[255]+y[256];
y[258]=-(lambda*x2*y[33]*y[44]*y[137]*y[194]);
y[259]=lambda*x2*y[33]*y[44]*y[128]*y[204];
y[260]=y[198]+y[258]+y[259];
y[261]=lambda*x2*y[33]*y[44]*y[128]*y[215];
y[262]=y[172]*y[197];
y[263]=y[261]+y[262];
y[264]=lambda*x2*y[33]*y[36]*y[44]*y[215];
y[265]=lambda*x2*y[33]*y[44]*y[128]*y[183];
y[266]=-(y[137]*y[172]);
y[267]=y[265]+y[266];
y[268]=-(lambda*x2*y[33]*y[36]*y[44]*y[201]);
y[269]=lambda*x2*y[33]*y[36]*y[44]*y[204];
y[270]=lambda*x2*y[33]*y[44]*y[187]*y[194];
y[271]=y[268]+y[269]+y[270];
y[272]=lambda*x2*y[33]*y[44]*y[180]*y[194];
y[273]=-(y[172]*y[201]);
y[274]=y[264]+y[272]+y[273];
y[275]=lambda*x2*y[33]*y[44]*y[183]*y[194];
y[276]=-(y[172]*y[204]);
y[277]=y[264]+y[275]+y[276];
y[278]=-(lambda*x2*y[33]*y[36]*y[44]*y[180]);
y[279]=lambda*x2*y[33]*y[36]*y[44]*y[183];
y[280]=y[172]*y[187];
y[281]=y[278]+y[279]+y[280];
y[282]=-x0;
y[283]=1.+y[282];
y[284]=2.*x0*x2*x4*y[1];
y[285]=2.*x1*x2*x4*y[1];
y[286]=x4*y[1]*y[16];
y[287]=2.*x0*x3*x4*y[1];
y[288]=2.*x1*x3*x4*y[1];
y[289]=2.*x2*x3*x4*y[1];
y[290]=x4*y[1]*y[20];
y[291]=-(x2*x3*x4*y[2]);
y[292]=-(x2*x4*y[11]);
y[293]=-(x3*y[31]);
y[294]=y[3]+y[4]+y[5]+y[6]+y[7]+y[8]+y[9]+y[10]+y[12]+y[13]+y[17]+y[284]+y[2\
85]+y[286]+y[287]+y[288]+y[289]+y[290]+y[291]+y[292]+y[293];
y[295]=-y[11];
y[296]=x4*y[1]*y[30];
y[297]=2.*x0*x1*x4*y[1];
y[298]=x4*y[1]*y[43];
y[299]=-(lambda*MYI*x0*y[21]*y[283]*y[294]);
y[300]=-x1;
y[301]=1.+y[300];
y[302]=y[3]+y[4]+y[5]+y[6]+y[7]+y[8]+y[9]+y[10]+y[12]+y[13]+y[17]+y[26]+y[28\
4]+y[285]+y[286]+y[287]+y[288]+y[289]+y[290]+y[291]+y[292]+y[295];
y[303]=-(lambda*MYI*x1*y[32]*y[301]*y[302]);
y[304]=-x4;
y[305]=1.+y[304];
y[306]=x0*y[1];
y[307]=x1*y[1];
y[308]=x2*y[1]*y[30];
y[309]=2.*x0*x1*x2*y[1];
y[310]=x2*y[1]*y[43];
y[311]=x0*y[1]*y[16];
y[312]=x1*y[1]*y[16];
y[313]=x3*y[1]*y[30];
y[314]=2.*x0*x1*x3*y[1];
y[315]=x3*y[1]*y[43];
y[316]=2.*x0*x2*x3*y[1];
y[317]=2.*x1*x2*x3*y[1];
y[318]=x0*y[1]*y[20];
y[319]=x1*y[1]*y[20];
y[320]=-(x0*x2*x3*y[2]);
y[321]=-(x1*x2*x3*y[2]);
y[322]=-(x0*x2*y[11]);
y[323]=-(x1*x2*y[11]);
y[324]=y[14]+y[15]+y[18]+y[19]+y[45]+y[46]+y[47]+y[124]+y[125]+y[306]+y[307]\
+y[308]+y[309]+y[310]+y[311]+y[312]+y[313]+y[314]+y[315]+y[316]+y[317]+y[31\
8]+y[319]+y[320]+y[321]+y[322]+y[323];
y[325]=-(lambda*MYI*x4*y[209]*y[305]*y[324]);
y[326]=x4+y[325];
y[327]=x0+y[299];
y[328]=-x2;
y[329]=1.+y[328];
y[330]=-(x0*x3*x4*y[2]);
y[331]=-(x1*x3*x4*y[2]);
y[332]=-(x0*x4*y[11]);
y[333]=-(x1*x4*y[11]);
y[334]=y[3]+y[4]+y[5]+y[6]+y[7]+y[13]+y[17]+y[26]+y[284]+y[285]+y[287]+y[288\
]+y[295]+y[296]+y[297]+y[298]+y[330]+y[331]+y[332]+y[333];
y[335]=-(lambda*MYI*x2*y[44]*y[329]*y[334]);
y[336]=x1+y[303];
y[337]=x2+y[335];
y[338]=-x3;
y[339]=1.+y[338];
y[340]=-(x2*y[2]);
y[341]=-(x0*x2*x4*y[2]);
y[342]=-(x1*x2*x4*y[2]);
y[343]=-(x0*y[31]);
y[344]=y[3]+y[4]+y[5]+y[6]+y[7]+y[13]+y[17]+y[123]+y[174]+y[192]+y[193]+y[28\
4]+y[285]+y[287]+y[288]+y[296]+y[297]+y[298]+y[340]+y[341]+y[342]+y[343];
y[345]=-(lambda*MYI*x3*y[132]*y[339]*y[344]);
y[346]=x3+y[345];
y[347]=pow(y[327],2);
y[348]=pow(y[336],2);
y[349]=pow(y[337],2);
y[350]=pow(y[346],2);
FOUT=MYI*x0*pow(y[1]+2.*y[1]*y[327]+y[1]*y[326]*y[327]+2.*y[1]*y[336]-y[11]*\
y[336]+y[1]*y[326]*y[336]+2.*y[1]*y[327]*y[336]+2.*y[1]*y[326]*y[327]*y[336\
]+2.*y[1]*y[337]-y[11]*y[337]+2.*y[1]*y[327]*y[337]+2.*y[1]*y[326]*y[327]*y\
[337]-y[11]*y[326]*y[327]*y[337]+2.*y[1]*y[336]*y[337]+2.*y[1]*y[326]*y[336\
]*y[337]-y[11]*y[326]*y[336]*y[337]+2.*y[1]*y[326]*y[327]*y[336]*y[337]+2.*\
y[1]*y[346]-y[2]*y[346]+2.*y[1]*y[327]*y[346]-y[31]*y[327]*y[346]+2.*y[1]*y\
[326]*y[327]*y[346]-y[2]*y[326]*y[327]*y[346]+2.*y[1]*y[336]*y[346]-y[2]*y[\
336]*y[346]+2.*y[1]*y[326]*y[336]*y[346]-y[2]*y[326]*y[336]*y[346]+2.*y[1]*\
y[326]*y[327]*y[336]*y[346]+2.*y[1]*y[337]*y[346]-y[2]*y[337]*y[346]+2.*y[1\
]*y[326]*y[327]*y[337]*y[346]-y[2]*y[326]*y[327]*y[337]*y[346]+2.*y[1]*y[32\
6]*y[336]*y[337]*y[346]-y[2]*y[326]*y[336]*y[337]*y[346]+y[1]*y[347]+y[1]*y\
[326]*y[347]+y[1]*y[326]*y[337]*y[347]+y[1]*y[326]*y[346]*y[347]+y[1]*y[348\
]+y[1]*y[326]*y[348]+y[1]*y[326]*y[337]*y[348]+y[1]*y[326]*y[346]*y[348]+y[\
1]*y[349]+y[1]*y[326]*y[327]*y[349]+y[1]*y[326]*y[336]*y[349]+y[1]*y[350]+y\
[1]*y[326]*y[327]*y[350]+y[1]*y[326]*y[336]*y[350],-3)*(lambda*x4*y[128]*y[\
188]*y[209]*(lambda*x3*y[122]*y[132]*y[189]*y[212]-y[208]*y[257]-lambda*x3*\
y[122]*y[132]*y[191]*y[260]+lambda*x3*y[122]*y[132]*y[177]*y[271])-lambda*x\
4*y[29]*y[188]*y[209]*(lambda*x3*y[122]*y[132]*y[194]*y[212]-y[218]*y[257]-\
lambda*x3*y[122]*y[132]*y[191]*y[263]+lambda*x3*y[122]*y[132]*y[177]*y[274]\
)+lambda*x4*y[29]*y[188]*y[209]*(lambda*x3*y[122]*y[132]*y[194]*y[260]-lamb\
da*x3*y[122]*y[132]*y[189]*y[263]-y[257]*y[267]+lambda*x3*y[122]*y[132]*y[1\
77]*y[277])-lambda*x4*y[177]*y[188]*y[209]*(-(lambda*x3*y[122]*y[132]*y[194\
]*y[208])+lambda*x3*y[122]*y[132]*y[189]*y[218]-lambda*x3*y[122]*y[132]*y[1\
91]*y[267]+lambda*x3*y[122]*y[132]*y[177]*y[281])+(y[42]-lambda*x0*y[1]*y[2\
09]-lambda*x1*y[1]*y[209]-2.*lambda*x0*x1*y[1]*y[209]-2.*lambda*x0*x2*y[1]*\
y[209]-2.*lambda*x1*x2*y[1]*y[209]-2.*lambda*x0*x1*x2*y[1]*y[209]-2.*lambda\
*x0*x3*y[1]*y[209]-2.*lambda*x1*x3*y[1]*y[209]-2.*lambda*x0*x1*x3*y[1]*y[20\
9]-2.*lambda*x0*x2*x3*y[1]*y[209]-2.*lambda*x1*x2*x3*y[1]*y[209]+2.*lambda*\
x0*x4*y[1]*y[209]+2.*lambda*x1*x4*y[1]*y[209]+4.*lambda*x0*x1*x4*y[1]*y[209\
]+4.*lambda*x0*x2*x4*y[1]*y[209]+4.*lambda*x1*x2*x4*y[1]*y[209]+4.*lambda*x\
0*x1*x2*x4*y[1]*y[209]+4.*lambda*x0*x3*x4*y[1]*y[209]+4.*lambda*x1*x3*x4*y[\
1]*y[209]+4.*lambda*x0*x1*x3*x4*y[1]*y[209]+4.*lambda*x0*x2*x3*x4*y[1]*y[20\
9]+4.*lambda*x1*x2*x3*x4*y[1]*y[209]+lambda*x0*x3*y[2]*y[209]+lambda*x1*x3*\
y[2]*y[209]+lambda*x0*x2*x3*y[2]*y[209]+lambda*x1*x2*x3*y[2]*y[209]-2.*lamb\
da*x0*x3*x4*y[2]*y[209]-2.*lambda*x1*x3*x4*y[2]*y[209]-2.*lambda*x0*x2*x3*x\
4*y[2]*y[209]-2.*lambda*x1*x2*x3*x4*y[2]*y[209]+lambda*x0*x2*y[11]*y[209]+l\
ambda*x1*x2*y[11]*y[209]-2.*lambda*x0*x2*x4*y[11]*y[209]-2.*lambda*x1*x2*x4\
*y[11]*y[209]-lambda*x0*y[1]*y[16]*y[209]-lambda*x1*y[1]*y[16]*y[209]+2.*la\
mbda*x0*x4*y[1]*y[16]*y[209]+2.*lambda*x1*x4*y[1]*y[16]*y[209]-lambda*x0*y[\
1]*y[20]*y[209]-lambda*x1*y[1]*y[20]*y[209]+2.*lambda*x0*x4*y[1]*y[20]*y[20\
9]+2.*lambda*x1*x4*y[1]*y[20]*y[209]-lambda*y[1]*y[30]*y[209]-lambda*x2*y[1\
]*y[30]*y[209]-lambda*x3*y[1]*y[30]*y[209]+2.*lambda*x4*y[1]*y[30]*y[209]+2\
.*lambda*x2*x4*y[1]*y[30]*y[209]+2.*lambda*x3*x4*y[1]*y[30]*y[209]-lambda*y\
[1]*y[43]*y[209]-lambda*x2*y[1]*y[43]*y[209]-lambda*x3*y[1]*y[43]*y[209]+2.\
*lambda*x4*y[1]*y[43]*y[209]+2.*lambda*x2*x4*y[1]*y[43]*y[209]+2.*lambda*x3\
*x4*y[1]*y[43]*y[209])*(-(lambda*x3*y[122]*y[132]*y[194]*y[271])+lambda*x3*\
y[122]*y[132]*y[189]*y[274]-lambda*x3*y[122]*y[132]*y[191]*y[277]+y[257]*y[\
281]))*(1.-lambda*MYI*y[21]*y[283]*y[294])*(1.+x0+x1+x2+x3+y[299]+y[303]+y[\
326]*y[327]+y[335]+y[326]*y[336]+y[326]*y[327]*y[337]+y[326]*y[336]*y[337]+\
y[345]+y[326]*y[327]*y[346]+y[326]*y[336]*y[346]);
return (FOUT);
    }
} ReduzeF1L2_021111010ord0f3;

int main() {

    const unsigned int MAXVAR = 5;

    // fit function to reduce variance
    integrators::Qmc<dcmplx,double,MAXVAR,integrators::transforms::Korobov<3>::type,integrators::fitfunctions::PolySingular::type> fitter;
    integrators::fitfunctions::PolySingularTransform<ReduzeF1L2_021111010ord0f3_t,double,MAXVAR> fitted_ReduzeF1L2_021111010ord0f3 = fitter.fit(ReduzeF1L2_021111010ord0f3);

    // setup integrator
    integrators::Qmc<dcmplx,double,MAXVAR,integrators::transforms::Korobov<3>::type> integrator;
    integrator.minm = 20;
    integrator.maxeval = 1; // do not iterate

    std::cout << "# n m Re[I] Im[I] Re[Abs. Err.] Im[Abs. Err.]" << std::endl;
    std::cout << std::setprecision(16);
    for(const auto& generating_vector : integrator.generatingvectors)
    {
        integrator.minn = generating_vector.first;
        integrators::result<dcmplx> result = integrator.integrate(fitted_ReduzeF1L2_021111010ord0f3);

        std::cout
        << result.n
        << " " << result.m
        << " " << result.integral.real()
        << " " << result.integral.imag()
        << " " << result.error.real()
        << " " << result.error.imag()
        << std::endl;
    }
}
