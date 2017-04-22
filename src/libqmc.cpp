#include <complex>

#include "qmc.cpp"
#include "qmc_generating_vectors.cpp"

// Instantiate Common Types
template class integrators::Qmc< std::complex<double>, double >;
template class integrators::Qmc< double , double >;
