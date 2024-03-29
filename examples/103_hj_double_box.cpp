/*
 * Compile without GPU support:
 *   c++ -std=c++17 -pthread -I../src 103_hj_double_box.cpp -o 103_hj_double_box.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++17 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 103_hj_double_box.cpp -o 103_hj_double_box.out -lgsl -lgslcblas
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

struct ReduzeF1L2_011112101ord1f18_t{
const unsigned long long int number_of_integration_variables  =6;
#ifdef __CUDACC__
    __host__ __device__
#endif
dcmplx myLog(dcmplx myarg) const {
   if (myarg.imag()==0.) myarg=dcmplx(myarg.real(),-0.0);
    return log(myarg);
}

#ifdef __CUDACC__
    __host__ __device__

#endif
dcmplx operator()(const double x[]) const
    {
double x0=x[0];
double x1=x[1];
double x2=x[2];
double x3=x[3];
double x4=x[4];
double x5=x[5];

//run 1111  //  s/mTs = 4./0.052 = 76.8
double p[] = {4,-3.73134,0.0520703,0.0271671,1,0.298679,0.228394,0.227807,0.564963,0.234168,0.227191,1};

double esx[2];
double em[2];
double lrs[6];
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
lrs[5]=p[10];
double bi=p[11];
dcmplx FOUT;
dcmplx MYI(0.,1.);
dcmplx y[891];
y[1]=em[0];
y[2]=pow(x4,2);
y[3]=pow(x5,2);
y[4]=em[1];
y[5]=esx[0];
y[6]=2.*x3*y[1];
y[7]=2.*x0*x3*y[1];
y[8]=2.*x1*x3*y[1];
y[9]=2.*x2*x3*y[1];
y[10]=2.*x4*y[1];
y[11]=2.*x1*x4*y[1];
y[12]=2.*x2*x4*y[1];
y[13]=2.*x5*y[1];
y[14]=2.*x1*x5*y[1];
y[15]=2.*x2*x5*y[1];
y[16]=pow(x0,2);
y[17]=2.*x0*y[1];
y[18]=2.*x0*x1*y[1];
y[19]=2.*x0*x2*y[1];
y[20]=-(x0*x1*y[5]);
y[21]=2.*y[1];
y[22]=2.*x1*y[1];
y[23]=2.*x2*y[1];
y[24]=2.*x0*x4*y[1];
y[25]=2.*x0*x5*y[1];
y[26]=4.*x0*x3*x4*y[1];
y[27]=4.*x0*x3*x5*y[1];
y[28]=-(x2*x3*y[5]);
y[29]=-(x4*y[5]);
y[30]=-(x2*x4*y[5]);
y[31]=2.*x3*x4*y[1];
y[32]=y[1]*y[2];
y[33]=2.*x3*x5*y[1];
y[34]=2.*x4*x5*y[1];
y[35]=y[1]*y[3];
y[36]=-(x5*y[4]);
y[37]=-(x4*x5*y[5]);
y[38]=lrs[0];
y[39]=lrs[1];
y[40]=-1.+x1;
y[41]=-(x2*y[5]);
y[42]=esx[1];
y[43]=-1.+x2;
y[44]=pow(lambda,2);
y[45]=-1.+x0;
y[46]=-(x0*y[4]);
y[47]=-(x0*x4*y[5]);
y[48]=y[7]+y[17]+y[21]+y[22]+y[23]+y[24]+y[25]+y[41]+y[46]+y[47];
y[49]=-(x3*y[4]);
y[50]=-(x3*x4*y[5]);
y[51]=y[1]+y[6]+y[10]+y[13]+y[31]+y[32]+y[33]+y[34]+y[35]+y[36]+y[37]+y[49]+\
y[50];
y[52]=4.*x0*x3*y[1];
y[53]=-y[4];
y[54]=-(x1*y[4]);
y[55]=-(x2*y[4]);
y[56]=-2.*x0*x3*y[4];
y[57]=-(x1*x4*y[5]);
y[58]=-2.*x0*x3*x4*y[5];
y[59]=y[6]+y[8]+y[9]+y[10]+y[11]+y[12]+y[13]+y[14]+y[15]+y[21]+y[22]+y[23]+y\
[26]+y[27]+y[28]+y[29]+y[30]+y[52]+y[53]+y[54]+y[55]+y[56]+y[57]+y[58];
y[60]=-MYI;
y[61]=pow(x1,2);
y[62]=lrs[2];
y[63]=-(x1*y[5]);
y[64]=-(lambda*y[1]*y[38]);
y[65]=2.*lambda*x0*y[1]*y[38];
y[66]=-(lambda*x1*y[1]*y[38]);
y[67]=2.*lambda*x0*x1*y[1]*y[38];
y[68]=-(lambda*x2*y[1]*y[38]);
y[69]=2.*lambda*x0*x2*y[1]*y[38];
y[70]=-2.*lambda*x3*y[1]*y[38];
y[71]=6.*lambda*x3*y[1]*y[16]*y[38];
y[72]=-2.*lambda*x1*x3*y[1]*y[38];
y[73]=4.*lambda*x0*x1*x3*y[1]*y[38];
y[74]=-2.*lambda*x2*x3*y[1]*y[38];
y[75]=4.*lambda*x0*x2*x3*y[1]*y[38];
y[76]=-2.*lambda*x4*y[1]*y[38];
y[77]=4.*lambda*x0*x4*y[1]*y[38];
y[78]=-2.*lambda*x1*x4*y[1]*y[38];
y[79]=4.*lambda*x0*x1*x4*y[1]*y[38];
y[80]=-2.*lambda*x2*x4*y[1]*y[38];
y[81]=4.*lambda*x0*x2*x4*y[1]*y[38];
y[82]=-2.*lambda*x3*x4*y[1]*y[38];
y[83]=-4.*lambda*x0*x3*x4*y[1]*y[38];
y[84]=12.*lambda*x3*x4*y[1]*y[16]*y[38];
y[85]=-2.*lambda*x1*x3*x4*y[1]*y[38];
y[86]=4.*lambda*x0*x1*x3*x4*y[1]*y[38];
y[87]=-2.*lambda*x2*x3*x4*y[1]*y[38];
y[88]=4.*lambda*x0*x2*x3*x4*y[1]*y[38];
y[89]=-(lambda*y[1]*y[2]*y[38]);
y[90]=2.*lambda*x0*y[1]*y[2]*y[38];
y[91]=-(lambda*x1*y[1]*y[2]*y[38]);
y[92]=2.*lambda*x0*x1*y[1]*y[2]*y[38];
y[93]=-(lambda*x2*y[1]*y[2]*y[38]);
y[94]=2.*lambda*x0*x2*y[1]*y[2]*y[38];
y[95]=-4.*lambda*x0*x3*y[1]*y[2]*y[38];
y[96]=6.*lambda*x3*y[1]*y[2]*y[16]*y[38];
y[97]=-2.*lambda*x5*y[1]*y[38];
y[98]=4.*lambda*x0*x5*y[1]*y[38];
y[99]=-2.*lambda*x1*x5*y[1]*y[38];
y[100]=4.*lambda*x0*x1*x5*y[1]*y[38];
y[101]=-2.*lambda*x2*x5*y[1]*y[38];
y[102]=4.*lambda*x0*x2*x5*y[1]*y[38];
y[103]=-2.*lambda*x3*x5*y[1]*y[38];
y[104]=-4.*lambda*x0*x3*x5*y[1]*y[38];
y[105]=12.*lambda*x3*x5*y[1]*y[16]*y[38];
y[106]=-2.*lambda*x1*x3*x5*y[1]*y[38];
y[107]=4.*lambda*x0*x1*x3*x5*y[1]*y[38];
y[108]=-2.*lambda*x2*x3*x5*y[1]*y[38];
y[109]=4.*lambda*x0*x2*x3*x5*y[1]*y[38];
y[110]=-2.*lambda*x4*x5*y[1]*y[38];
y[111]=4.*lambda*x0*x4*x5*y[1]*y[38];
y[112]=-2.*lambda*x1*x4*x5*y[1]*y[38];
y[113]=4.*lambda*x0*x1*x4*x5*y[1]*y[38];
y[114]=-2.*lambda*x2*x4*x5*y[1]*y[38];
y[115]=4.*lambda*x0*x2*x4*x5*y[1]*y[38];
y[116]=-8.*lambda*x0*x3*x4*x5*y[1]*y[38];
y[117]=12.*lambda*x3*x4*x5*y[1]*y[16]*y[38];
y[118]=-(lambda*y[1]*y[3]*y[38]);
y[119]=2.*lambda*x0*y[1]*y[3]*y[38];
y[120]=-(lambda*x1*y[1]*y[3]*y[38]);
y[121]=2.*lambda*x0*x1*y[1]*y[3]*y[38];
y[122]=-(lambda*x2*y[1]*y[3]*y[38]);
y[123]=2.*lambda*x0*x2*y[1]*y[3]*y[38];
y[124]=-4.*lambda*x0*x3*y[1]*y[3]*y[38];
y[125]=6.*lambda*x3*y[1]*y[3]*y[16]*y[38];
y[126]=lambda*x1*x3*y[4]*y[38];
y[127]=-2.*lambda*x0*x1*x3*y[4]*y[38];
y[128]=lambda*x5*y[4]*y[38];
y[129]=-2.*lambda*x0*x5*y[4]*y[38];
y[130]=lambda*x1*x5*y[4]*y[38];
y[131]=-2.*lambda*x0*x1*x5*y[4]*y[38];
y[132]=lambda*x2*x5*y[4]*y[38];
y[133]=-2.*lambda*x0*x2*x5*y[4]*y[38];
y[134]=4.*lambda*x0*x3*x5*y[4]*y[38];
y[135]=-6.*lambda*x3*x5*y[4]*y[16]*y[38];
y[136]=lambda*x1*x3*x4*y[5]*y[38];
y[137]=-2.*lambda*x0*x1*x3*x4*y[5]*y[38];
y[138]=lambda*x2*x3*x5*y[5]*y[38];
y[139]=-2.*lambda*x0*x2*x3*x5*y[5]*y[38];
y[140]=lambda*x4*x5*y[5]*y[38];
y[141]=-2.*lambda*x0*x4*x5*y[5]*y[38];
y[142]=lambda*x1*x4*x5*y[5]*y[38];
y[143]=-2.*lambda*x0*x1*x4*x5*y[5]*y[38];
y[144]=lambda*x2*x4*x5*y[5]*y[38];
y[145]=-2.*lambda*x0*x2*x4*x5*y[5]*y[38];
y[146]=4.*lambda*x0*x3*x4*x5*y[5]*y[38];
y[147]=-6.*lambda*x3*x4*x5*y[5]*y[16]*y[38];
y[148]=lambda*x3*y[38]*y[42];
y[149]=-2.*lambda*x0*x3*y[38]*y[42];
y[150]=y[60]+y[64]+y[65]+y[66]+y[67]+y[68]+y[69]+y[70]+y[71]+y[72]+y[73]+y[7\
4]+y[75]+y[76]+y[77]+y[78]+y[79]+y[80]+y[81]+y[82]+y[83]+y[84]+y[85]+y[86]+\
y[87]+y[88]+y[89]+y[90]+y[91]+y[92]+y[93]+y[94]+y[95]+y[96]+y[97]+y[98]+y[9\
9]+y[100]+y[101]+y[102]+y[103]+y[104]+y[105]+y[106]+y[107]+y[108]+y[109]+y[\
110]+y[111]+y[112]+y[113]+y[114]+y[115]+y[116]+y[117]+y[118]+y[119]+y[120]+\
y[121]+y[122]+y[123]+y[124]+y[125]+y[126]+y[127]+y[128]+y[129]+y[130]+y[131\
]+y[132]+y[133]+y[134]+y[135]+y[136]+y[137]+y[138]+y[139]+y[140]+y[141]+y[1\
42]+y[143]+y[144]+y[145]+y[146]+y[147]+y[148]+y[149];
y[151]=-2.*lambda*y[1]*y[39];
y[152]=-(lambda*x0*y[1]*y[39]);
y[153]=2.*lambda*x0*x1*y[1]*y[39];
y[154]=6.*lambda*y[1]*y[39]*y[61];
y[155]=-2.*lambda*x2*y[1]*y[39];
y[156]=4.*lambda*x1*x2*y[1]*y[39];
y[157]=-2.*lambda*x3*y[1]*y[39];
y[158]=-2.*lambda*x0*x3*y[1]*y[39];
y[159]=4.*lambda*x0*x1*x3*y[1]*y[39];
y[160]=6.*lambda*x3*y[1]*y[39]*y[61];
y[161]=-2.*lambda*x2*x3*y[1]*y[39];
y[162]=4.*lambda*x1*x2*x3*y[1]*y[39];
y[163]=-2.*lambda*x4*y[1]*y[39];
y[164]=-2.*lambda*x0*x4*y[1]*y[39];
y[165]=4.*lambda*x0*x1*x4*y[1]*y[39];
y[166]=6.*lambda*x4*y[1]*y[39]*y[61];
y[167]=-2.*lambda*x2*x4*y[1]*y[39];
y[168]=4.*lambda*x1*x2*x4*y[1]*y[39];
y[169]=-2.*lambda*x0*x3*x4*y[1]*y[39];
y[170]=4.*lambda*x0*x1*x3*x4*y[1]*y[39];
y[171]=-(lambda*x0*y[1]*y[2]*y[39]);
y[172]=2.*lambda*x0*x1*y[1]*y[2]*y[39];
y[173]=-2.*lambda*x5*y[1]*y[39];
y[174]=-2.*lambda*x0*x5*y[1]*y[39];
y[175]=4.*lambda*x0*x1*x5*y[1]*y[39];
y[176]=6.*lambda*x5*y[1]*y[39]*y[61];
y[177]=-2.*lambda*x2*x5*y[1]*y[39];
y[178]=4.*lambda*x1*x2*x5*y[1]*y[39];
y[179]=-2.*lambda*x0*x3*x5*y[1]*y[39];
y[180]=4.*lambda*x0*x1*x3*x5*y[1]*y[39];
y[181]=-2.*lambda*x0*x4*x5*y[1]*y[39];
y[182]=4.*lambda*x0*x1*x4*x5*y[1]*y[39];
y[183]=-(lambda*x0*y[1]*y[3]*y[39]);
y[184]=2.*lambda*x0*x1*y[1]*y[3]*y[39];
y[185]=lambda*x0*x3*y[4]*y[39];
y[186]=-2.*lambda*x0*x1*x3*y[4]*y[39];
y[187]=lambda*x0*x5*y[4]*y[39];
y[188]=-2.*lambda*x0*x1*x5*y[4]*y[39];
y[189]=lambda*x2*y[5]*y[39];
y[190]=-2.*lambda*x1*x2*y[5]*y[39];
y[191]=lambda*x2*x3*y[5]*y[39];
y[192]=-2.*lambda*x1*x2*x3*y[5]*y[39];
y[193]=lambda*x2*x4*y[5]*y[39];
y[194]=-2.*lambda*x1*x2*x4*y[5]*y[39];
y[195]=lambda*x0*x3*x4*y[5]*y[39];
y[196]=-2.*lambda*x0*x1*x3*x4*y[5]*y[39];
y[197]=lambda*x2*x5*y[5]*y[39];
y[198]=-2.*lambda*x1*x2*x5*y[5]*y[39];
y[199]=lambda*x0*x4*x5*y[5]*y[39];
y[200]=-2.*lambda*x0*x1*x4*x5*y[5]*y[39];
y[201]=y[60]+y[151]+y[152]+y[153]+y[154]+y[155]+y[156]+y[157]+y[158]+y[159]+\
y[160]+y[161]+y[162]+y[163]+y[164]+y[165]+y[166]+y[167]+y[168]+y[169]+y[170\
]+y[171]+y[172]+y[173]+y[174]+y[175]+y[176]+y[177]+y[178]+y[179]+y[180]+y[1\
81]+y[182]+y[183]+y[184]+y[185]+y[186]+y[187]+y[188]+y[189]+y[190]+y[191]+y\
[192]+y[193]+y[194]+y[195]+y[196]+y[197]+y[198]+y[199]+y[200];
y[202]=-1.+x3;
y[203]=-(x3*x5*y[5]);
y[204]=y[1]+y[6]+y[10]+y[13]+y[31]+y[32]+y[33]+y[34]+y[35]+y[36]+y[37]+y[203\
];
y[205]=-y[5];
y[206]=-(x3*y[5]);
y[207]=-(x5*y[5]);
y[208]=y[6]+y[10]+y[13]+y[21]+y[29]+y[205]+y[206]+y[207];
y[209]=-(x0*x3*y[5]);
y[210]=y[7]+y[17]+y[21]+y[22]+y[23]+y[24]+y[25]+y[46]+y[47]+y[63]+y[209];
y[211]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[59]);
y[212]=lambda*x1*y[39]*y[40]*y[48]*y[150];
y[213]=y[211]+y[212];
y[214]=pow(x2,2);
y[215]=lrs[3];
y[216]=-(x2*x5*y[5]);
y[217]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[59]*y[208]);
y[218]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[48]*y[204];
y[219]=y[217]+y[218];
y[220]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[48]*y[51];
y[221]=-(lambda*x0*y[38]*y[45]*y[59]*y[201]);
y[222]=y[220]+y[221];
y[223]=-2.*lambda*y[1]*y[62];
y[224]=-(lambda*x0*y[1]*y[62]);
y[225]=-2.*lambda*x1*y[1]*y[62];
y[226]=2.*lambda*x0*x2*y[1]*y[62];
y[227]=4.*lambda*x1*x2*y[1]*y[62];
y[228]=6.*lambda*y[1]*y[62]*y[214];
y[229]=-2.*lambda*x3*y[1]*y[62];
y[230]=-2.*lambda*x0*x3*y[1]*y[62];
y[231]=-2.*lambda*x1*x3*y[1]*y[62];
y[232]=4.*lambda*x0*x2*x3*y[1]*y[62];
y[233]=4.*lambda*x1*x2*x3*y[1]*y[62];
y[234]=6.*lambda*x3*y[1]*y[62]*y[214];
y[235]=-2.*lambda*x4*y[1]*y[62];
y[236]=-2.*lambda*x0*x4*y[1]*y[62];
y[237]=-2.*lambda*x1*x4*y[1]*y[62];
y[238]=4.*lambda*x0*x2*x4*y[1]*y[62];
y[239]=4.*lambda*x1*x2*x4*y[1]*y[62];
y[240]=6.*lambda*x4*y[1]*y[62]*y[214];
y[241]=-2.*lambda*x0*x3*x4*y[1]*y[62];
y[242]=4.*lambda*x0*x2*x3*x4*y[1]*y[62];
y[243]=-(lambda*x0*y[1]*y[2]*y[62]);
y[244]=2.*lambda*x0*x2*y[1]*y[2]*y[62];
y[245]=-2.*lambda*x5*y[1]*y[62];
y[246]=-2.*lambda*x0*x5*y[1]*y[62];
y[247]=-2.*lambda*x1*x5*y[1]*y[62];
y[248]=4.*lambda*x0*x2*x5*y[1]*y[62];
y[249]=4.*lambda*x1*x2*x5*y[1]*y[62];
y[250]=6.*lambda*x5*y[1]*y[62]*y[214];
y[251]=-2.*lambda*x0*x3*x5*y[1]*y[62];
y[252]=4.*lambda*x0*x2*x3*x5*y[1]*y[62];
y[253]=-2.*lambda*x0*x4*x5*y[1]*y[62];
y[254]=4.*lambda*x0*x2*x4*x5*y[1]*y[62];
y[255]=-(lambda*x0*y[1]*y[3]*y[62]);
y[256]=2.*lambda*x0*x2*y[1]*y[3]*y[62];
y[257]=lambda*x0*x5*y[4]*y[62];
y[258]=-2.*lambda*x0*x2*x5*y[4]*y[62];
y[259]=lambda*x1*y[5]*y[62];
y[260]=-2.*lambda*x1*x2*y[5]*y[62];
y[261]=lambda*x1*x3*y[5]*y[62];
y[262]=-2.*lambda*x1*x2*x3*y[5]*y[62];
y[263]=lambda*x1*x4*y[5]*y[62];
y[264]=-2.*lambda*x1*x2*x4*y[5]*y[62];
y[265]=lambda*x1*x5*y[5]*y[62];
y[266]=-2.*lambda*x1*x2*x5*y[5]*y[62];
y[267]=lambda*x0*x3*x5*y[5]*y[62];
y[268]=-2.*lambda*x0*x2*x3*x5*y[5]*y[62];
y[269]=lambda*x0*x4*x5*y[5]*y[62];
y[270]=-2.*lambda*x0*x2*x4*x5*y[5]*y[62];
y[271]=y[60]+y[223]+y[224]+y[225]+y[226]+y[227]+y[228]+y[229]+y[230]+y[231]+\
y[232]+y[233]+y[234]+y[235]+y[236]+y[237]+y[238]+y[239]+y[240]+y[241]+y[242\
]+y[243]+y[244]+y[245]+y[246]+y[247]+y[248]+y[249]+y[250]+y[251]+y[252]+y[2\
53]+y[254]+y[255]+y[256]+y[257]+y[258]+y[259]+y[260]+y[261]+y[262]+y[263]+y\
[264]+y[265]+y[266]+y[267]+y[268]+y[269]+y[270];
y[272]=2.*y[1]*y[16];
y[273]=2.*x4*y[1]*y[16];
y[274]=2.*x5*y[1]*y[16];
y[275]=-(x0*x2*y[5]);
y[276]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[204]);
y[277]=lambda*x1*y[39]*y[40]*y[150]*y[208];
y[278]=y[276]+y[277];
y[279]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[208];
y[280]=-(lambda*x0*y[38]*y[45]*y[201]*y[204]);
y[281]=y[279]+y[280];
y[282]=pow(y[51],2);
y[283]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[282]);
y[284]=y[150]*y[201];
y[285]=y[283]+y[284];
y[286]=-1.+x4;
y[287]=-(x0*x5*y[5]);
y[288]=y[17]+y[21]+y[22]+y[23]+y[24]+y[25]+y[41]+y[46]+y[47];
y[289]=4.*x0*x4*y[1];
y[290]=2.*x0*y[1]*y[2];
y[291]=4.*x0*x5*y[1];
y[292]=4.*x0*x4*x5*y[1];
y[293]=2.*x0*y[1]*y[3];
y[294]=-2.*x0*x5*y[4];
y[295]=-2.*x0*x4*x5*y[5];
y[296]=-y[42];
y[297]=y[10]+y[11]+y[12]+y[13]+y[14]+y[15]+y[17]+y[21]+y[22]+y[23]+y[54]+y[5\
7]+y[216]+y[289]+y[290]+y[291]+y[292]+y[293]+y[294]+y[295]+y[296];
y[298]=y[17]+y[21]+y[22]+y[23]+y[24]+y[25]+y[63]+y[287];
y[299]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[59]*y[288]);
y[300]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[48]*y[297];
y[301]=y[299]+y[300];
y[302]=-(y[4]*y[16]);
y[303]=-(x4*y[5]*y[16]);
y[304]=y[17]+y[18]+y[19]+y[272]+y[273]+y[274]+y[275]+y[302]+y[303];
y[305]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[297]);
y[306]=lambda*x1*y[39]*y[40]*y[150]*y[288];
y[307]=y[305]+y[306];
y[308]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[288];
y[309]=-(lambda*x0*y[38]*y[45]*y[201]*y[297]);
y[310]=y[308]+y[309];
y[311]=-(lambda*x2*y[43]*y[62]*y[208]*y[213]);
y[312]=lambda*x2*y[43]*y[62]*y[204]*y[222];
y[313]=lambda*x2*y[43]*y[62]*y[210]*y[285];
y[314]=y[311]+y[312]+y[313];
y[315]=lrs[4];
y[316]=lambda*x2*y[43]*y[62]*y[204]*y[301];
y[317]=lambda*x2*y[43]*y[62]*y[210]*y[307];
y[318]=-(lambda*x2*y[43]*y[62]*y[213]*y[298]);
y[319]=y[316]+y[317]+y[318];
y[320]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[204]*y[288];
y[321]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[208]*y[297]);
y[322]=y[320]+y[321];
y[323]=lambda*x2*y[43]*y[62]*y[204]*y[219];
y[324]=lambda*x2*y[43]*y[62]*y[210]*y[278];
y[325]=-(y[213]*y[271]);
y[326]=y[323]+y[324]+y[325];
y[327]=-(lambda*y[1]*y[215]);
y[328]=-2.*lambda*x0*y[1]*y[215];
y[329]=-(lambda*y[1]*y[16]*y[215]);
y[330]=-2.*lambda*x1*y[1]*y[215];
y[331]=-2.*lambda*x0*x1*y[1]*y[215];
y[332]=-(lambda*y[1]*y[61]*y[215]);
y[333]=-2.*lambda*x2*y[1]*y[215];
y[334]=-2.*lambda*x0*x2*y[1]*y[215];
y[335]=-2.*lambda*x1*x2*y[1]*y[215];
y[336]=-(lambda*y[1]*y[214]*y[215]);
y[337]=2.*lambda*x3*y[1]*y[215];
y[338]=4.*lambda*x0*x3*y[1]*y[215];
y[339]=2.*lambda*x3*y[1]*y[16]*y[215];
y[340]=4.*lambda*x1*x3*y[1]*y[215];
y[341]=4.*lambda*x0*x1*x3*y[1]*y[215];
y[342]=2.*lambda*x3*y[1]*y[61]*y[215];
y[343]=4.*lambda*x2*x3*y[1]*y[215];
y[344]=4.*lambda*x0*x2*x3*y[1]*y[215];
y[345]=4.*lambda*x1*x2*x3*y[1]*y[215];
y[346]=2.*lambda*x3*y[1]*y[214]*y[215];
y[347]=-2.*lambda*x0*x4*y[1]*y[215];
y[348]=-2.*lambda*x4*y[1]*y[16]*y[215];
y[349]=-2.*lambda*x0*x1*x4*y[1]*y[215];
y[350]=-2.*lambda*x0*x2*x4*y[1]*y[215];
y[351]=4.*lambda*x0*x3*x4*y[1]*y[215];
y[352]=4.*lambda*x3*x4*y[1]*y[16]*y[215];
y[353]=4.*lambda*x0*x1*x3*x4*y[1]*y[215];
y[354]=4.*lambda*x0*x2*x3*x4*y[1]*y[215];
y[355]=-(lambda*y[1]*y[2]*y[16]*y[215]);
y[356]=2.*lambda*x3*y[1]*y[2]*y[16]*y[215];
y[357]=-2.*lambda*x0*x5*y[1]*y[215];
y[358]=-2.*lambda*x5*y[1]*y[16]*y[215];
y[359]=-2.*lambda*x0*x1*x5*y[1]*y[215];
y[360]=-2.*lambda*x0*x2*x5*y[1]*y[215];
y[361]=4.*lambda*x0*x3*x5*y[1]*y[215];
y[362]=4.*lambda*x3*x5*y[1]*y[16]*y[215];
y[363]=4.*lambda*x0*x1*x3*x5*y[1]*y[215];
y[364]=4.*lambda*x0*x2*x3*x5*y[1]*y[215];
y[365]=-2.*lambda*x4*x5*y[1]*y[16]*y[215];
y[366]=4.*lambda*x3*x4*x5*y[1]*y[16]*y[215];
y[367]=-(lambda*y[1]*y[3]*y[16]*y[215]);
y[368]=2.*lambda*x3*y[1]*y[3]*y[16]*y[215];
y[369]=lambda*x0*x1*y[4]*y[215];
y[370]=-2.*lambda*x0*x1*x3*y[4]*y[215];
y[371]=lambda*x5*y[4]*y[16]*y[215];
y[372]=-2.*lambda*x3*x5*y[4]*y[16]*y[215];
y[373]=lambda*x1*x2*y[5]*y[215];
y[374]=-2.*lambda*x1*x2*x3*y[5]*y[215];
y[375]=lambda*x0*x1*x4*y[5]*y[215];
y[376]=-2.*lambda*x0*x1*x3*x4*y[5]*y[215];
y[377]=lambda*x0*x2*x5*y[5]*y[215];
y[378]=-2.*lambda*x0*x2*x3*x5*y[5]*y[215];
y[379]=lambda*x4*x5*y[5]*y[16]*y[215];
y[380]=-2.*lambda*x3*x4*x5*y[5]*y[16]*y[215];
y[381]=lambda*x0*y[42]*y[215];
y[382]=-2.*lambda*x0*x3*y[42]*y[215];
y[383]=y[60]+y[327]+y[328]+y[329]+y[330]+y[331]+y[332]+y[333]+y[334]+y[335]+\
y[336]+y[337]+y[338]+y[339]+y[340]+y[341]+y[342]+y[343]+y[344]+y[345]+y[346\
]+y[347]+y[348]+y[349]+y[350]+y[351]+y[352]+y[353]+y[354]+y[355]+y[356]+y[3\
57]+y[358]+y[359]+y[360]+y[361]+y[362]+y[363]+y[364]+y[365]+y[366]+y[367]+y\
[368]+y[369]+y[370]+y[371]+y[372]+y[373]+y[374]+y[375]+y[376]+y[377]+y[378]\
+y[379]+y[380]+y[381]+y[382];
y[384]=lambda*x2*y[43]*y[62]*y[208]*y[301];
y[385]=-(lambda*x2*y[43]*y[62]*y[222]*y[298]);
y[386]=lambda*x2*y[43]*y[62]*y[210]*y[310];
y[387]=y[384]+y[385]+y[386];
y[388]=-(lambda*x2*y[43]*y[62]*y[219]*y[298]);
y[389]=lambda*x2*y[43]*y[62]*y[210]*y[322];
y[390]=y[271]*y[301];
y[391]=y[388]+y[389]+y[390];
y[392]=lambda*x2*y[43]*y[62]*y[208]*y[219];
y[393]=lambda*x2*y[43]*y[62]*y[210]*y[281];
y[394]=-(y[222]*y[271]);
y[395]=y[392]+y[393]+y[394];
y[396]=2.*x3*y[1]*y[16];
y[397]=-(x0*y[5]);
y[398]=-(x3*y[5]*y[16]);
y[399]=y[17]+y[18]+y[19]+y[20]+y[275]+y[396]+y[397]+y[398];
y[400]=-(lambda*x2*y[43]*y[62]*y[208]*y[307]);
y[401]=lambda*x2*y[43]*y[62]*y[204]*y[310];
y[402]=lambda*x2*y[43]*y[62]*y[285]*y[298];
y[403]=y[400]+y[401]+y[402];
y[404]=lambda*x2*y[43]*y[62]*y[204]*y[322];
y[405]=lambda*x2*y[43]*y[62]*y[278]*y[298];
y[406]=-(y[271]*y[307]);
y[407]=y[404]+y[405]+y[406];
y[408]=lambda*x2*y[43]*y[62]*y[208]*y[322];
y[409]=lambda*x2*y[43]*y[62]*y[281]*y[298];
y[410]=-(y[271]*y[310]);
y[411]=y[408]+y[409]+y[410];
y[412]=-(lambda*x2*y[43]*y[62]*y[208]*y[278]);
y[413]=lambda*x2*y[43]*y[62]*y[204]*y[281];
y[414]=y[271]*y[285];
y[415]=y[412]+y[413]+y[414];
y[416]=-1.+x5;
y[417]=y[7]+y[17]+y[21]+y[22]+y[23]+y[24]+y[25]+y[63]+y[287];
y[418]=y[7]+y[17]+y[21]+y[22]+y[23]+y[24]+y[25]+y[41]+y[209]+y[287];
y[419]=-(x1*x3*y[5]);
y[420]=-(x1*x5*y[5]);
y[421]=-2.*x0*x3*x5*y[5];
y[422]=y[6]+y[8]+y[9]+y[10]+y[11]+y[12]+y[13]+y[14]+y[15]+y[21]+y[22]+y[23]+\
y[26]+y[27]+y[52]+y[207]+y[216]+y[419]+y[420]+y[421];
y[423]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[59]*y[418]);
y[424]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[48]*y[422];
y[425]=y[423]+y[424];
y[426]=-(x5*y[5]*y[16]);
y[427]=y[17]+y[18]+y[19]+y[20]+y[272]+y[273]+y[274]+y[426];
y[428]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[422]);
y[429]=lambda*x1*y[39]*y[40]*y[150]*y[418];
y[430]=y[428]+y[429];
y[431]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[51]*y[418];
y[432]=-(lambda*x0*y[38]*y[45]*y[201]*y[422]);
y[433]=y[431]+y[432];
y[434]=lambda*x2*y[43]*y[62]*y[204]*y[425];
y[435]=-(lambda*x2*y[43]*y[62]*y[213]*y[417]);
y[436]=lambda*x2*y[43]*y[62]*y[210]*y[430];
y[437]=y[434]+y[435]+y[436];
y[438]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[208]*y[422]);
y[439]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[204]*y[418];
y[440]=y[438]+y[439];
y[441]=lambda*x2*y[43]*y[62]*y[208]*y[425];
y[442]=-(lambda*x2*y[43]*y[62]*y[222]*y[417]);
y[443]=lambda*x2*y[43]*y[62]*y[210]*y[433];
y[444]=y[441]+y[442]+y[443];
y[445]=-(lambda*x2*y[43]*y[62]*y[219]*y[417]);
y[446]=lambda*x2*y[43]*y[62]*y[210]*y[440];
y[447]=y[271]*y[425];
y[448]=y[445]+y[446]+y[447];
y[449]=-(lambda*x2*y[43]*y[62]*y[208]*y[430]);
y[450]=lambda*x2*y[43]*y[62]*y[204]*y[433];
y[451]=lambda*x2*y[43]*y[62]*y[285]*y[417];
y[452]=y[449]+y[450]+y[451];
y[453]=lambda*x2*y[43]*y[62]*y[204]*y[440];
y[454]=lambda*x2*y[43]*y[62]*y[278]*y[417];
y[455]=-(y[271]*y[430]);
y[456]=y[453]+y[454]+y[455];
y[457]=lambda*x2*y[43]*y[62]*y[208]*y[440];
y[458]=lambda*x2*y[43]*y[62]*y[281]*y[417];
y[459]=-(y[271]*y[433]);
y[460]=y[457]+y[458]+y[459];
y[461]=-(lambda*x3*y[202]*y[215]*y[298]*y[314]);
y[462]=lambda*x3*y[202]*y[215]*y[288]*y[326];
y[463]=-(lambda*x3*y[202]*y[215]*y[297]*y[395]);
y[464]=lambda*x3*y[202]*y[215]*y[304]*y[415];
y[465]=y[461]+y[462]+y[463]+y[464];
y[466]=lrs[5];
y[467]=lambda*x3*y[202]*y[215]*y[288]*y[437];
y[468]=-(lambda*x3*y[202]*y[215]*y[297]*y[444]);
y[469]=-(lambda*x3*y[202]*y[215]*y[314]*y[427]);
y[470]=lambda*x3*y[202]*y[215]*y[304]*y[452];
y[471]=y[467]+y[468]+y[469]+y[470];
y[472]=-(x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[288]*y[422]);
y[473]=x0*x1*y[38]*y[39]*y[40]*y[44]*y[45]*y[297]*y[418];
y[474]=y[472]+y[473];
y[475]=lambda*x2*y[43]*y[62]*y[298]*y[425];
y[476]=-(lambda*x2*y[43]*y[62]*y[301]*y[417]);
y[477]=lambda*x2*y[43]*y[62]*y[210]*y[474];
y[478]=y[475]+y[476]+y[477];
y[479]=lambda*x2*y[43]*y[62]*y[204]*y[474];
y[480]=lambda*x2*y[43]*y[62]*y[307]*y[417];
y[481]=-(lambda*x2*y[43]*y[62]*y[298]*y[430]);
y[482]=y[479]+y[480]+y[481];
y[483]=lambda*x2*y[43]*y[62]*y[208]*y[474];
y[484]=-(lambda*x2*y[43]*y[62]*y[298]*y[433]);
y[485]=lambda*x2*y[43]*y[62]*y[310]*y[417];
y[486]=y[483]+y[484]+y[485];
y[487]=lambda*x3*y[202]*y[215]*y[288]*y[319];
y[488]=-(lambda*x3*y[202]*y[215]*y[297]*y[387]);
y[489]=lambda*x3*y[202]*y[215]*y[304]*y[403];
y[490]=-(y[314]*y[383]);
y[491]=y[487]+y[488]+y[489]+y[490];
y[492]=-(lambda*y[1]*y[315]);
y[493]=-2.*lambda*x0*y[1]*y[315];
y[494]=-2.*lambda*x1*y[1]*y[315];
y[495]=-2.*lambda*x0*x1*y[1]*y[315];
y[496]=-(lambda*y[1]*y[61]*y[315]);
y[497]=-2.*lambda*x2*y[1]*y[315];
y[498]=-2.*lambda*x0*x2*y[1]*y[315];
y[499]=-2.*lambda*x1*x2*y[1]*y[315];
y[500]=-(lambda*y[1]*y[214]*y[315]);
y[501]=-2.*lambda*x0*x3*y[1]*y[315];
y[502]=-2.*lambda*x3*y[1]*y[16]*y[315];
y[503]=-2.*lambda*x0*x1*x3*y[1]*y[315];
y[504]=-2.*lambda*x0*x2*x3*y[1]*y[315];
y[505]=2.*lambda*x4*y[1]*y[315];
y[506]=4.*lambda*x1*x4*y[1]*y[315];
y[507]=2.*lambda*x4*y[1]*y[61]*y[315];
y[508]=4.*lambda*x2*x4*y[1]*y[315];
y[509]=4.*lambda*x1*x2*x4*y[1]*y[315];
y[510]=2.*lambda*x4*y[1]*y[214]*y[315];
y[511]=4.*lambda*x0*x3*x4*y[1]*y[315];
y[512]=4.*lambda*x0*x1*x3*x4*y[1]*y[315];
y[513]=4.*lambda*x0*x2*x3*x4*y[1]*y[315];
y[514]=6.*lambda*x0*y[1]*y[2]*y[315];
y[515]=6.*lambda*x0*x1*y[1]*y[2]*y[315];
y[516]=6.*lambda*x0*x2*y[1]*y[2]*y[315];
y[517]=6.*lambda*x3*y[1]*y[2]*y[16]*y[315];
y[518]=-2.*lambda*x0*x5*y[1]*y[315];
y[519]=-2.*lambda*x0*x1*x5*y[1]*y[315];
y[520]=-2.*lambda*x0*x2*x5*y[1]*y[315];
y[521]=-2.*lambda*x3*x5*y[1]*y[16]*y[315];
y[522]=4.*lambda*x0*x4*x5*y[1]*y[315];
y[523]=4.*lambda*x0*x1*x4*x5*y[1]*y[315];
y[524]=4.*lambda*x0*x2*x4*x5*y[1]*y[315];
y[525]=4.*lambda*x3*x4*x5*y[1]*y[16]*y[315];
y[526]=lambda*x1*x2*y[5]*y[315];
y[527]=lambda*x0*x1*x3*y[5]*y[315];
y[528]=-2.*lambda*x1*x2*x4*y[5]*y[315];
y[529]=-2.*lambda*x0*x1*x3*x4*y[5]*y[315];
y[530]=lambda*x0*x5*y[5]*y[315];
y[531]=lambda*x0*x1*x5*y[5]*y[315];
y[532]=lambda*x0*x2*x5*y[5]*y[315];
y[533]=lambda*x3*x5*y[5]*y[16]*y[315];
y[534]=-2.*lambda*x0*x4*x5*y[5]*y[315];
y[535]=-2.*lambda*x0*x1*x4*x5*y[5]*y[315];
y[536]=-2.*lambda*x0*x2*x4*x5*y[5]*y[315];
y[537]=-2.*lambda*x3*x4*x5*y[5]*y[16]*y[315];
y[538]=y[60]+y[492]+y[493]+y[494]+y[495]+y[496]+y[497]+y[498]+y[499]+y[500]+\
y[501]+y[502]+y[503]+y[504]+y[505]+y[506]+y[507]+y[508]+y[509]+y[510]+y[511\
]+y[512]+y[513]+y[514]+y[515]+y[516]+y[517]+y[518]+y[519]+y[520]+y[521]+y[5\
22]+y[523]+y[524]+y[525]+y[526]+y[527]+y[528]+y[529]+y[530]+y[531]+y[532]+y\
[533]+y[534]+y[535]+y[536]+y[537];
y[539]=lambda*x3*y[202]*y[215]*y[298]*y[437];
y[540]=-(lambda*x3*y[202]*y[215]*y[297]*y[448]);
y[541]=-(lambda*x3*y[202]*y[215]*y[326]*y[427]);
y[542]=lambda*x3*y[202]*y[215]*y[304]*y[456];
y[543]=y[539]+y[540]+y[541]+y[542];
y[544]=-(lambda*x3*y[202]*y[215]*y[297]*y[478]);
y[545]=-(lambda*x3*y[202]*y[215]*y[319]*y[427]);
y[546]=lambda*x3*y[202]*y[215]*y[304]*y[482];
y[547]=y[383]*y[437];
y[548]=y[544]+y[545]+y[546]+y[547];
y[549]=-(lambda*x2*y[43]*y[62]*y[298]*y[440]);
y[550]=lambda*x2*y[43]*y[62]*y[322]*y[417];
y[551]=y[271]*y[474];
y[552]=y[549]+y[550]+y[551];
y[553]=lambda*x3*y[202]*y[215]*y[298]*y[319];
y[554]=-(lambda*x3*y[202]*y[215]*y[297]*y[391]);
y[555]=lambda*x3*y[202]*y[215]*y[304]*y[407];
y[556]=-(y[326]*y[383]);
y[557]=y[553]+y[554]+y[555]+y[556];
y[558]=lambda*x3*y[202]*y[215]*y[298]*y[444];
y[559]=-(lambda*x3*y[202]*y[215]*y[288]*y[448]);
y[560]=-(lambda*x3*y[202]*y[215]*y[395]*y[427]);
y[561]=lambda*x3*y[202]*y[215]*y[304]*y[460];
y[562]=y[558]+y[559]+y[560]+y[561];
y[563]=-(lambda*x3*y[202]*y[215]*y[288]*y[478]);
y[564]=-(lambda*x3*y[202]*y[215]*y[387]*y[427]);
y[565]=lambda*x3*y[202]*y[215]*y[304]*y[486];
y[566]=y[383]*y[444];
y[567]=y[563]+y[564]+y[565]+y[566];
y[568]=-(lambda*x3*y[202]*y[215]*y[298]*y[478]);
y[569]=-(lambda*x3*y[202]*y[215]*y[391]*y[427]);
y[570]=lambda*x3*y[202]*y[215]*y[304]*y[552];
y[571]=y[383]*y[448];
y[572]=y[568]+y[569]+y[570]+y[571];
y[573]=lambda*x3*y[202]*y[215]*y[298]*y[387];
y[574]=-(lambda*x3*y[202]*y[215]*y[288]*y[391]);
y[575]=lambda*x3*y[202]*y[215]*y[304]*y[411];
y[576]=-(y[383]*y[395]);
y[577]=y[573]+y[574]+y[575]+y[576];
y[578]=-(lambda*x3*y[202]*y[215]*y[298]*y[452]);
y[579]=lambda*x3*y[202]*y[215]*y[288]*y[456];
y[580]=-(lambda*x3*y[202]*y[215]*y[297]*y[460]);
y[581]=lambda*x3*y[202]*y[215]*y[415]*y[427];
y[582]=y[578]+y[579]+y[580]+y[581];
y[583]=lambda*x3*y[202]*y[215]*y[288]*y[482];
y[584]=-(lambda*x3*y[202]*y[215]*y[297]*y[486]);
y[585]=lambda*x3*y[202]*y[215]*y[403]*y[427];
y[586]=-(y[383]*y[452]);
y[587]=y[583]+y[584]+y[585]+y[586];
y[588]=lambda*x3*y[202]*y[215]*y[298]*y[482];
y[589]=-(lambda*x3*y[202]*y[215]*y[297]*y[552]);
y[590]=lambda*x3*y[202]*y[215]*y[407]*y[427];
y[591]=-(y[383]*y[456]);
y[592]=y[588]+y[589]+y[590]+y[591];
y[593]=lambda*x3*y[202]*y[215]*y[298]*y[486];
y[594]=-(lambda*x3*y[202]*y[215]*y[288]*y[552]);
y[595]=lambda*x3*y[202]*y[215]*y[411]*y[427];
y[596]=-(y[383]*y[460]);
y[597]=y[593]+y[594]+y[595]+y[596];
y[598]=-(lambda*x3*y[202]*y[215]*y[298]*y[403]);
y[599]=lambda*x3*y[202]*y[215]*y[288]*y[407];
y[600]=-(lambda*x3*y[202]*y[215]*y[297]*y[411]);
y[601]=y[383]*y[415];
y[602]=y[598]+y[599]+y[600]+y[601];
y[603]=-x1;
y[604]=1.+y[603];
y[605]=x0*y[1];
y[606]=2.*x0*x3*x4*y[1];
y[607]=x0*y[1]*y[2];
y[608]=2.*x0*x3*x5*y[1];
y[609]=2.*x0*x4*x5*y[1];
y[610]=x0*y[1]*y[3];
y[611]=-(x0*x3*y[4]);
y[612]=-(x0*x5*y[4]);
y[613]=-(x0*x3*x4*y[5]);
y[614]=-(x0*x4*x5*y[5]);
y[615]=y[6]+y[7]+y[8]+y[9]+y[10]+y[11]+y[12]+y[13]+y[14]+y[15]+y[21]+y[22]+y\
[23]+y[24]+y[25]+y[28]+y[30]+y[41]+y[216]+y[605]+y[606]+y[607]+y[608]+y[609\
]+y[610]+y[611]+y[612]+y[613]+y[614];
y[616]=-x0;
y[617]=1.+y[616];
y[618]=x1*y[1];
y[619]=x2*y[1];
y[620]=2.*x1*x3*x4*y[1];
y[621]=2.*x2*x3*x4*y[1];
y[622]=x1*y[1]*y[2];
y[623]=x2*y[1]*y[2];
y[624]=2.*x0*x3*y[1]*y[2];
y[625]=2.*x1*x3*x5*y[1];
y[626]=2.*x2*x3*x5*y[1];
y[627]=2.*x1*x4*x5*y[1];
y[628]=2.*x2*x4*x5*y[1];
y[629]=4.*x0*x3*x4*x5*y[1];
y[630]=x1*y[1]*y[3];
y[631]=x2*y[1]*y[3];
y[632]=2.*x0*x3*y[1]*y[3];
y[633]=-(x1*x3*y[4]);
y[634]=-(x1*x5*y[4]);
y[635]=-(x2*x5*y[4]);
y[636]=-2.*x0*x3*x5*y[4];
y[637]=-(x1*x3*x4*y[5]);
y[638]=-(x2*x3*x5*y[5]);
y[639]=-(x1*x4*x5*y[5]);
y[640]=-(x2*x4*x5*y[5]);
y[641]=-2.*x0*x3*x4*x5*y[5];
y[642]=-(x3*y[42]);
y[643]=y[1]+y[6]+y[7]+y[8]+y[9]+y[10]+y[11]+y[12]+y[13]+y[14]+y[15]+y[26]+y[\
27]+y[31]+y[32]+y[33]+y[34]+y[35]+y[36]+y[37]+y[618]+y[619]+y[620]+y[621]+y\
[622]+y[623]+y[624]+y[625]+y[626]+y[627]+y[628]+y[629]+y[630]+y[631]+y[632]\
+y[633]+y[634]+y[635]+y[636]+y[637]+y[638]+y[639]+y[640]+y[641]+y[642];
y[644]=-x3;
y[645]=1.+y[644];
y[646]=y[1]*y[16];
y[647]=y[1]*y[61];
y[648]=2.*x1*x2*y[1];
y[649]=y[1]*y[214];
y[650]=2.*x0*x1*x4*y[1];
y[651]=2.*x0*x2*x4*y[1];
y[652]=y[1]*y[2]*y[16];
y[653]=2.*x0*x1*x5*y[1];
y[654]=2.*x0*x2*x5*y[1];
y[655]=2.*x4*x5*y[1]*y[16];
y[656]=y[1]*y[3]*y[16];
y[657]=-(x0*x1*y[4]);
y[658]=-(x5*y[4]*y[16]);
y[659]=-(x1*x2*y[5]);
y[660]=-(x0*x1*x4*y[5]);
y[661]=-(x0*x2*x5*y[5]);
y[662]=-(x4*x5*y[5]*y[16]);
y[663]=-(x0*y[42]);
y[664]=y[1]+y[17]+y[18]+y[19]+y[22]+y[23]+y[24]+y[25]+y[273]+y[274]+y[646]+y\
[647]+y[648]+y[649]+y[650]+y[651]+y[652]+y[653]+y[654]+y[655]+y[656]+y[657]\
+y[658]+y[659]+y[660]+y[661]+y[662]+y[663];
y[665]=-(lambda*MYI*x3*y[215]*y[645]*y[664]);
y[666]=-(lambda*MYI*x1*y[39]*y[604]*y[615]);
y[667]=x3+y[665];
y[668]=-x2;
y[669]=1.+y[668];
y[670]=-(x0*x3*x5*y[5]);
y[671]=y[6]+y[7]+y[8]+y[9]+y[10]+y[11]+y[12]+y[13]+y[14]+y[15]+y[21]+y[22]+y\
[23]+y[24]+y[25]+y[57]+y[63]+y[419]+y[420]+y[605]+y[606]+y[607]+y[608]+y[60\
9]+y[610]+y[612]+y[614]+y[670];
y[672]=-(lambda*MYI*x2*y[62]*y[669]*y[671]);
y[673]=x1+y[666];
y[674]=-x4;
y[675]=1.+y[674];
y[676]=2.*x0*x1*x3*y[1];
y[677]=2.*x0*x2*x3*y[1];
y[678]=2.*x3*x4*y[1]*y[16];
y[679]=2.*x3*x5*y[1]*y[16];
y[680]=-(x0*x1*x3*y[5]);
y[681]=-(x0*x1*x5*y[5]);
y[682]=-(x3*x5*y[5]*y[16]);
y[683]=y[1]+y[7]+y[17]+y[18]+y[19]+y[22]+y[23]+y[24]+y[25]+y[287]+y[396]+y[6\
47]+y[648]+y[649]+y[650]+y[651]+y[653]+y[654]+y[659]+y[661]+y[676]+y[677]+y\
[678]+y[679]+y[680]+y[681]+y[682];
y[684]=-(lambda*MYI*x4*y[315]*y[675]*y[683]);
y[685]=x2+y[672];
y[686]=x4+y[684];
y[687]=-(lambda*MYI*x0*y[38]*y[617]*y[643]);
y[688]=x0+y[687];
y[689]=-x5;
y[690]=1.+y[689];
y[691]=-(x0*x2*y[4]);
y[692]=-(x3*y[4]*y[16]);
y[693]=-(x0*x2*x3*y[5]);
y[694]=-(x0*x2*x4*y[5]);
y[695]=-(x3*x4*y[5]*y[16]);
y[696]=y[1]+y[7]+y[17]+y[18]+y[19]+y[22]+y[23]+y[24]+y[25]+y[46]+y[47]+y[396\
]+y[647]+y[648]+y[649]+y[650]+y[651]+y[653]+y[654]+y[657]+y[659]+y[660]+y[6\
76]+y[677]+y[678]+y[679]+y[691]+y[692]+y[693]+y[694]+y[695];
y[697]=-(lambda*MYI*x5*y[466]*y[690]*y[696]);
y[698]=x5+y[697];
y[699]=pow(y[673],2);
y[700]=pow(y[685],2);
y[701]=pow(y[688],2);
y[702]=pow(y[686],2);
y[703]=pow(y[698],2);
y[704]=-(lambda*MYI*y[38]*y[617]*y[643]);
y[705]=1.+y[704];
y[706]=-(lambda*MYI*y[39]*y[604]*y[615]);
y[707]=1.+y[706];
y[708]=-(lambda*x4*y[286]*y[315]*y[427]*y[465]);
y[709]=lambda*x4*y[286]*y[315]*y[417]*y[491];
y[710]=-(lambda*x4*y[286]*y[315]*y[418]*y[557]);
y[711]=lambda*x4*y[286]*y[315]*y[422]*y[577];
y[712]=lambda*x4*y[286]*y[315]*y[399]*y[602];
y[713]=y[708]+y[709]+y[710]+y[711]+y[712];
y[714]=lambda*x5*y[399]*y[416]*y[466]*y[713];
y[715]=lambda*x4*y[286]*y[315]*y[417]*y[471];
y[716]=-(lambda*x4*y[286]*y[315]*y[418]*y[543]);
y[717]=lambda*x4*y[286]*y[315]*y[422]*y[562];
y[718]=lambda*x4*y[286]*y[315]*y[399]*y[582];
y[719]=-(y[465]*y[538]);
y[720]=y[715]+y[716]+y[717]+y[718]+y[719];
y[721]=-(lambda*x5*y[304]*y[416]*y[466]*y[720]);
y[722]=lambda*x4*y[286]*y[315]*y[427]*y[471];
y[723]=-(lambda*x4*y[286]*y[315]*y[418]*y[548]);
y[724]=lambda*x4*y[286]*y[315]*y[422]*y[567];
y[725]=lambda*x4*y[286]*y[315]*y[399]*y[587];
y[726]=-(y[491]*y[538]);
y[727]=y[722]+y[723]+y[724]+y[725]+y[726];
y[728]=lambda*x5*y[210]*y[416]*y[466]*y[727];
y[729]=lambda*x4*y[286]*y[315]*y[427]*y[543];
y[730]=-(lambda*x4*y[286]*y[315]*y[417]*y[548]);
y[731]=lambda*x4*y[286]*y[315]*y[422]*y[572];
y[732]=lambda*x4*y[286]*y[315]*y[399]*y[592];
y[733]=-(y[538]*y[557]);
y[734]=y[729]+y[730]+y[731]+y[732]+y[733];
y[735]=-(lambda*x5*y[48]*y[416]*y[466]*y[734]);
y[736]=lambda*x4*y[286]*y[315]*y[427]*y[562];
y[737]=-(lambda*x4*y[286]*y[315]*y[417]*y[567]);
y[738]=lambda*x4*y[286]*y[315]*y[418]*y[572];
y[739]=lambda*x4*y[286]*y[315]*y[399]*y[597];
y[740]=-(y[538]*y[577]);
y[741]=y[736]+y[737]+y[738]+y[739]+y[740];
y[742]=lambda*x5*y[59]*y[416]*y[466]*y[741];
y[743]=-(lambda*x4*y[286]*y[315]*y[427]*y[582]);
y[744]=lambda*x4*y[286]*y[315]*y[417]*y[587];
y[745]=-(lambda*x4*y[286]*y[315]*y[418]*y[592]);
y[746]=lambda*x4*y[286]*y[315]*y[422]*y[597];
y[747]=y[538]*y[602];
y[748]=y[743]+y[744]+y[745]+y[746]+y[747];
y[749]=-(lambda*y[1]*y[466]);
y[750]=-2.*lambda*x0*y[1]*y[466];
y[751]=-2.*lambda*x1*y[1]*y[466];
y[752]=-2.*lambda*x0*x1*y[1]*y[466];
y[753]=-(lambda*y[1]*y[61]*y[466]);
y[754]=-2.*lambda*x2*y[1]*y[466];
y[755]=-2.*lambda*x0*x2*y[1]*y[466];
y[756]=-2.*lambda*x1*x2*y[1]*y[466];
y[757]=-(lambda*y[1]*y[214]*y[466]);
y[758]=-2.*lambda*x0*x3*y[1]*y[466];
y[759]=-2.*lambda*x3*y[1]*y[16]*y[466];
y[760]=-2.*lambda*x0*x1*x3*y[1]*y[466];
y[761]=-2.*lambda*x0*x2*x3*y[1]*y[466];
y[762]=-2.*lambda*x0*x4*y[1]*y[466];
y[763]=-2.*lambda*x0*x1*x4*y[1]*y[466];
y[764]=-2.*lambda*x0*x2*x4*y[1]*y[466];
y[765]=-2.*lambda*x3*x4*y[1]*y[16]*y[466];
y[766]=2.*lambda*x5*y[1]*y[466];
y[767]=4.*lambda*x1*x5*y[1]*y[466];
y[768]=2.*lambda*x5*y[1]*y[61]*y[466];
y[769]=4.*lambda*x2*x5*y[1]*y[466];
y[770]=4.*lambda*x1*x2*x5*y[1]*y[466];
y[771]=2.*lambda*x5*y[1]*y[214]*y[466];
y[772]=4.*lambda*x0*x3*x5*y[1]*y[466];
y[773]=4.*lambda*x0*x1*x3*x5*y[1]*y[466];
y[774]=4.*lambda*x0*x2*x3*x5*y[1]*y[466];
y[775]=4.*lambda*x0*x4*x5*y[1]*y[466];
y[776]=4.*lambda*x0*x1*x4*x5*y[1]*y[466];
y[777]=4.*lambda*x0*x2*x4*x5*y[1]*y[466];
y[778]=4.*lambda*x3*x4*x5*y[1]*y[16]*y[466];
y[779]=6.*lambda*x0*y[1]*y[3]*y[466];
y[780]=6.*lambda*x0*x1*y[1]*y[3]*y[466];
y[781]=6.*lambda*x0*x2*y[1]*y[3]*y[466];
y[782]=6.*lambda*x3*y[1]*y[3]*y[16]*y[466];
y[783]=lambda*x0*y[4]*y[466];
y[784]=lambda*x0*x1*y[4]*y[466];
y[785]=lambda*x0*x2*y[4]*y[466];
y[786]=lambda*x3*y[4]*y[16]*y[466];
y[787]=-2.*lambda*x0*x5*y[4]*y[466];
y[788]=-2.*lambda*x0*x1*x5*y[4]*y[466];
y[789]=-2.*lambda*x0*x2*x5*y[4]*y[466];
y[790]=-2.*lambda*x3*x5*y[4]*y[16]*y[466];
y[791]=lambda*x1*x2*y[5]*y[466];
y[792]=lambda*x0*x2*x3*y[5]*y[466];
y[793]=lambda*x0*x4*y[5]*y[466];
y[794]=lambda*x0*x1*x4*y[5]*y[466];
y[795]=lambda*x0*x2*x4*y[5]*y[466];
y[796]=lambda*x3*x4*y[5]*y[16]*y[466];
y[797]=-2.*lambda*x1*x2*x5*y[5]*y[466];
y[798]=-2.*lambda*x0*x2*x3*x5*y[5]*y[466];
y[799]=-2.*lambda*x0*x4*x5*y[5]*y[466];
y[800]=-2.*lambda*x0*x1*x4*x5*y[5]*y[466];
y[801]=-2.*lambda*x0*x2*x4*x5*y[5]*y[466];
y[802]=-2.*lambda*x3*x4*x5*y[5]*y[16]*y[466];
y[803]=y[60]+y[749]+y[750]+y[751]+y[752]+y[753]+y[754]+y[755]+y[756]+y[757]+\
y[758]+y[759]+y[760]+y[761]+y[762]+y[763]+y[764]+y[765]+y[766]+y[767]+y[768\
]+y[769]+y[770]+y[771]+y[772]+y[773]+y[774]+y[775]+y[776]+y[777]+y[778]+y[7\
79]+y[780]+y[781]+y[782]+y[783]+y[784]+y[785]+y[786]+y[787]+y[788]+y[789]+y\
[790]+y[791]+y[792]+y[793]+y[794]+y[795]+y[796]+y[797]+y[798]+y[799]+y[800]\
+y[801]+y[802];
y[804]=-(y[748]*y[803]);
y[805]=y[714]+y[721]+y[728]+y[735]+y[742]+y[804];
y[806]=y[667]*y[688];
y[807]=y[667]*y[673];
y[808]=y[667]*y[685];
y[809]=y[673]*y[686];
y[810]=y[685]*y[686];
y[811]=y[667]*y[686]*y[688];
y[812]=y[673]*y[698];
y[813]=y[685]*y[698];
y[814]=y[667]*y[688]*y[698];
y[815]=1.+x1+x2+x3+x4+x5+y[665]+y[666]+y[672]+y[684]+y[697]+y[806]+y[807]+y[\
808]+y[809]+y[810]+y[811]+y[812]+y[813]+y[814];
y[816]=pow(y[815],2);
y[817]=y[1]*y[688];
y[818]=2.*y[1]*y[673];
y[819]=y[1]*y[673]*y[688];
y[820]=y[1]*y[699];
y[821]=2.*y[1]*y[685];
y[822]=y[1]*y[685]*y[688];
y[823]=2.*y[1]*y[673]*y[685];
y[824]=-(y[5]*y[673]*y[685]);
y[825]=y[1]*y[700];
y[826]=y[1]*y[667];
y[827]=2.*y[1]*y[667]*y[688];
y[828]=-(y[42]*y[667]*y[688]);
y[829]=y[1]*y[667]*y[701];
y[830]=2.*y[1]*y[667]*y[673];
y[831]=2.*y[1]*y[667]*y[673]*y[688];
y[832]=-(y[4]*y[667]*y[673]*y[688]);
y[833]=y[1]*y[667]*y[699];
y[834]=2.*y[1]*y[667]*y[685];
y[835]=2.*y[1]*y[667]*y[685]*y[688];
y[836]=2.*y[1]*y[667]*y[673]*y[685];
y[837]=-(y[5]*y[667]*y[673]*y[685]);
y[838]=y[1]*y[667]*y[700];
y[839]=y[1]*y[686];
y[840]=2.*y[1]*y[686]*y[688];
y[841]=2.*y[1]*y[673]*y[686];
y[842]=2.*y[1]*y[673]*y[686]*y[688];
y[843]=y[1]*y[686]*y[699];
y[844]=2.*y[1]*y[685]*y[686];
y[845]=2.*y[1]*y[685]*y[686]*y[688];
y[846]=2.*y[1]*y[673]*y[685]*y[686];
y[847]=-(y[5]*y[673]*y[685]*y[686]);
y[848]=y[1]*y[686]*y[700];
y[849]=2.*y[1]*y[667]*y[686]*y[688];
y[850]=2.*y[1]*y[667]*y[686]*y[701];
y[851]=2.*y[1]*y[667]*y[673]*y[686]*y[688];
y[852]=-(y[5]*y[667]*y[673]*y[686]*y[688]);
y[853]=2.*y[1]*y[667]*y[685]*y[686]*y[688];
y[854]=y[1]*y[688]*y[702];
y[855]=y[1]*y[673]*y[688]*y[702];
y[856]=y[1]*y[685]*y[688]*y[702];
y[857]=y[1]*y[667]*y[701]*y[702];
y[858]=y[1]*y[698];
y[859]=2.*y[1]*y[688]*y[698];
y[860]=-(y[4]*y[688]*y[698]);
y[861]=2.*y[1]*y[673]*y[698];
y[862]=2.*y[1]*y[673]*y[688]*y[698];
y[863]=-(y[4]*y[673]*y[688]*y[698]);
y[864]=y[1]*y[698]*y[699];
y[865]=2.*y[1]*y[685]*y[698];
y[866]=2.*y[1]*y[685]*y[688]*y[698];
y[867]=-(y[4]*y[685]*y[688]*y[698]);
y[868]=2.*y[1]*y[673]*y[685]*y[698];
y[869]=-(y[5]*y[673]*y[685]*y[698]);
y[870]=y[1]*y[698]*y[700];
y[871]=2.*y[1]*y[667]*y[688]*y[698];
y[872]=2.*y[1]*y[667]*y[698]*y[701];
y[873]=-(y[4]*y[667]*y[698]*y[701]);
y[874]=2.*y[1]*y[667]*y[673]*y[688]*y[698];
y[875]=2.*y[1]*y[667]*y[685]*y[688]*y[698];
y[876]=-(y[5]*y[667]*y[685]*y[688]*y[698]);
y[877]=2.*y[1]*y[686]*y[688]*y[698];
y[878]=-(y[5]*y[686]*y[688]*y[698]);
y[879]=2.*y[1]*y[673]*y[686]*y[688]*y[698];
y[880]=-(y[5]*y[673]*y[686]*y[688]*y[698]);
y[881]=2.*y[1]*y[685]*y[686]*y[688]*y[698];
y[882]=-(y[5]*y[685]*y[686]*y[688]*y[698]);
y[883]=2.*y[1]*y[667]*y[686]*y[698]*y[701];
y[884]=-(y[5]*y[667]*y[686]*y[698]*y[701]);
y[885]=y[1]*y[688]*y[703];
y[886]=y[1]*y[673]*y[688]*y[703];
y[887]=y[1]*y[685]*y[688]*y[703];
y[888]=y[1]*y[667]*y[701]*y[703];
y[889]=y[1]+y[817]+y[818]+y[819]+y[820]+y[821]+y[822]+y[823]+y[824]+y[825]+y\
[826]+y[827]+y[828]+y[829]+y[830]+y[831]+y[832]+y[833]+y[834]+y[835]+y[836]\
+y[837]+y[838]+y[839]+y[840]+y[841]+y[842]+y[843]+y[844]+y[845]+y[846]+y[84\
7]+y[848]+y[849]+y[850]+y[851]+y[852]+y[853]+y[854]+y[855]+y[856]+y[857]+y[\
858]+y[859]+y[860]+y[861]+y[862]+y[863]+y[864]+y[865]+y[866]+y[867]+y[868]+\
y[869]+y[870]+y[871]+y[872]+y[873]+y[874]+y[875]+y[876]+y[877]+y[878]+y[879\
]+y[880]+y[881]+y[882]+y[883]+y[884]+y[885]+y[886]+y[887]+y[888];
y[890]=pow(y[889],-4);
FOUT=x0*x1*myLog(x0)*y[705]*y[707]*y[805]*y[816]*y[890]+x0*x1*(myLog(y[705])\
*y[705]*y[707]*y[805]*y[816]*y[890]+3.*myLog(y[815])*y[705]*y[707]*y[805]*y\
[816]*y[890]-2.*myLog(y[889])*y[705]*y[707]*y[805]*y[816]*y[890]);
return (FOUT);
    }
} ReduzeF1L2_011112101ord1f18;

int main() {

    const unsigned int MAXVAR = 6;

    // fit function to reduce variance
    integrators::Qmc<dcmplx,double,MAXVAR,integrators::transforms::None::type,integrators::fitfunctions::PolySingular::type> fitter;
    integrators::fitfunctions::PolySingularTransform<ReduzeF1L2_011112101ord1f18_t,double,MAXVAR> fitted_ReduzeF1L2_011112101ord1f18 = fitter.fit(ReduzeF1L2_011112101ord1f18);

    // setup integrator
    integrators::Qmc<dcmplx,double,MAXVAR,integrators::transforms::Korobov<1>::type> integrator;
    integrator.minm = 20;
    integrator.maxeval = 1; // do not iterate

    std::cout << "# n m Re[I] Im[I] Re[Abs. Err.] Im[Abs. Err.]" << std::endl;
    std::cout << std::setprecision(16);
    for(const auto& generating_vector : integrator.generatingvectors)
    {
        integrator.minn = generating_vector.first;
        integrators::result<dcmplx> result = integrator.integrate( fitted_ReduzeF1L2_011112101ord1f18);

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
