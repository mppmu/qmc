#include <complex>

#include "qmc.hpp"

// Instantiate Common Types
template class integrators::Qmc< std::complex<double>, double >;
template class integrators::Qmc< double , double >;
template double qmcutil::mul_mod< double, double, unsigned long long int >(unsigned long long int, unsigned long long int, unsigned long long int);
template unsigned long long int qmcutil::mul_mod< unsigned long long int, double, unsigned long long int >(unsigned long long int, unsigned long long int, unsigned long long int);
