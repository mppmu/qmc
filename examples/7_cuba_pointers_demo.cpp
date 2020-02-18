/*
 * Compile without GPU support:
 *   c++ -std=c++11 -pthread -I../src 7_cuba_pointers_demo.cpp -o 7_cuba_pointers_demo.out -lgsl -lgslcblas
 * Compile with GPU support:
 *   nvcc -arch=<arch> -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants -I../src 7_cuba_pointers_demo.cpp -o 7_cuba_pointers_demo.out -lgsl -lgslcblas
 * Here `<arch>` is the architecture of the target GPU or `compute_30` if you are happy to use Just-in-Time compilation (See the Nvidia `nvcc` manual for more details).
 */

#include <iostream>
#include <cmath> // sin, cos, exp
#include "qmc.hpp"

struct cuda_error : public std::runtime_error { using std::runtime_error::runtime_error; };
#define CUDA_SAFE_CALL(err) { if (err != cudaSuccess) throw cuda_error(std::string(cudaGetErrorString(err)) + ": " + std::string(__FILE__) + " line " + std::to_string(__LINE__)); }

// Functions to integrate
#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif
HOSTDEVICE double func1(double x[]) { return sin(x[0])*cos(x[1])*exp(x[2]); }
HOSTDEVICE double func2(double x[]) { return 1./( (x[0] + x[1])*(x[0] + x[1]) + .003 )*cos(x[1])*exp(x[2]); }
HOSTDEVICE double func3(double x[]) { return 1./(3.75 - cos(M_PI*x[0]) - cos(M_PI*x[1]) - cos(M_PI*x[2])); }
HOSTDEVICE double func4(double x[]) { return fabs(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - .125); }
HOSTDEVICE double func5(double x[]) { return exp(-x[0]*x[0] - x[1]*x[1] - x[2]*x[2]); }
HOSTDEVICE double func6(double x[]) { return 1./(1. - x[0]*x[1]*x[2] + 1e-10); }
HOSTDEVICE double func7(double x[]) { return sqrt(fabs(x[0] - x[1] - x[2])); }
HOSTDEVICE double func8(double x[]) { return exp(-x[0]*x[1]*x[2]); }
HOSTDEVICE double func9(double x[]) { return x[0]*x[0]/(cos(x[0] + x[1] + x[2] + 1.) + 5.); }
HOSTDEVICE double func10(double x[]) { return ( (x[0] > .5) ? 1./sqrt(x[0]*x[1]*x[2] + 1e-5) : sqrt(x[0]*x[1]*x[2]) ); }
HOSTDEVICE double func11(double x[]) { return ( ((x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) < 1.) ? 1. : 0. ); }

// Function pointers on the device
// Note: must be declared in global scope
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif
typedef double integrand(double x[]);
DEVICE integrand* device_func1 = func1; // must be in global scope
DEVICE integrand* device_func2 = func2; // must be in global scope
DEVICE integrand* device_func3 = func3; // must be in global scope
DEVICE integrand* device_func4 = func4; // must be in global scope
DEVICE integrand* device_func5 = func5; // must be in global scope
DEVICE integrand* device_func6 = func6; // must be in global scope
DEVICE integrand* device_func7 = func7; // must be in global scope
DEVICE integrand* device_func8 = func8; // must be in global scope
DEVICE integrand* device_func9 = func9; // must be in global scope
DEVICE integrand* device_func10 = func10; // must be in global scope
DEVICE integrand* device_func11 = func11; // must be in global scope

// Functions for getting address of a device function on the host
// Note: must be declared in global scope and in same translation unit as device function pointers due to call of cudaMemcpyFromSymbol
typedef integrand*(get_device_address)();
integrand* get_device_address_func1()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func1, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func2()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func2, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func3()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func3, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func4()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func4, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func5()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func5, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func6()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func6, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func7()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func7, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func8()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func8, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func9()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func9, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func10()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func10, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}
integrand* get_device_address_func11()
{
#ifdef __CUDACC__
    integrand* device_address_on_host;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&device_address_on_host, device_func11, sizeof(integrand*))); // must be called by a function in global scope
    return device_address_on_host;
#else
    return nullptr;
#endif
}

// Generic functor for storing pointers to the functions
struct my_functor_t {

    const unsigned long long int number_of_integration_variables = 3;
    
    get_device_address* device_address_getter;

    integrand* host_address;
    integrand* device_address;

#ifdef __CUDACC__
    __host__ __device__
#endif
    double operator()(double* x) const
    {
#ifdef __CUDA_ARCH__
        return device_address(x);
#else
        return host_address(x);
#endif
    }

    //Copy constructor
    my_functor_t(const my_functor_t& my_functor):
    device_address_getter(my_functor.device_address_getter),
    host_address(my_functor.host_address),
    device_address(my_functor.device_address_getter()) // update device_address for current device
    {};

    // Constructor
    my_functor_t(integrand* host_address, get_device_address* device_address_getter):
    device_address_getter(device_address_getter),
    host_address(host_address),
    device_address(device_address_getter())
    {};

};

int main() {

    const unsigned int MAXVAR = 3;

    std::vector<my_functor_t> functors =
    {
        {*func1,*get_device_address_func1},
        {*func2,*get_device_address_func2},
        {*func3,*get_device_address_func3},
        {*func4,*get_device_address_func4},
        {*func5,*get_device_address_func5},
        {*func6,*get_device_address_func6},
        {*func7,*get_device_address_func7},
        {*func8,*get_device_address_func8},
        {*func9,*get_device_address_func9},
        {*func10,*get_device_address_func10},
        {*func11,*get_device_address_func11}
    };

    integrators::Qmc<double,double,MAXVAR,integrators::transforms::Korobov<3>::type> real_integrator;

    real_integrator.minn = 10000;
    real_integrator.maxeval = 1;

    integrators::result<double> real_result;
    for( const auto& functor : functors )
    {
        real_result = real_integrator.integrate(functor);
        std::cout << real_result.integral << " " << real_result.error << std::endl;
    }

}
