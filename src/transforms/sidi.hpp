#ifndef QMC_TRANSFORMS_SIDI_H
#define QMC_TRANSFORMS_SIDI_H

#include <type_traits> // enable_if, is_same

#include "detail/ipow.hpp"

namespace integrators
{
    namespace transforms
    {
        #ifdef __CUDACC__
            #define HOSTDEVICE __host__ __device__
        #else
            #define HOSTDEVICE
        #endif

        using std::sin;
        using std::cos;

        /*
         * Sidi Transform: Sidi<D,U,r>(func) takes the weight r Sidi transform of func
         */
        template<typename F1, typename D, typename U, U r, typename V = void>
        struct Sidi
        {
            static_assert(!std::is_same<V,V>::value, "Sidi transform with the chosen weight \"r\" is not implemented.");
        };

        template<typename F1, typename D, typename U, U r>
        struct Sidi<F1,D,U,r,typename std::enable_if<r==0>::type>
        {
            F1 f; // original function
            const U dim;

            Sidi(F1 f) : f(f), dim(f.dim) {};

            HOSTDEVICE
            auto operator()(D* x) -> decltype(f(x)) const
            {
                return f(x);
            }
        };

        #define SIDI(RANK,NORMALIZATION,...) \
            template<typename F1, typename D, typename U, U r> \
            struct Sidi<F1,D,U,r,typename std::enable_if<r==RANK>::type> \
            { \
                F1 f; /* original function */ \
                const U dim; \
                const D pi = acos( D(-1) ); \
 \
                Sidi(F1 f) : f(f), dim(f.dim) {}; \
 \
                HOSTDEVICE \
                auto operator()(D* x) -> decltype(f(x)) const \
                { \
                    D wgt = 1; \
                    for(U s = 0; s<dim; s++) \
                    { \
                        wgt *= (NORMALIZATION)*detail::IPow<D,U,r>::value( sin(pi*x[s]) ); \
                        __VA_ARGS__; \
                        /* loss of precision can cause x < 0 or x > 1 must keep in x \elem [0,1] */ \
                        if (x[s] > D(1)) x[s] = D(1); \
                        if (x[s] < D(0)) x[s] = D(0); \
                    } \
                    return wgt * f(x); \
                } \
            }

        SIDI( 1, pi/D(2), x[s]=(D(1)-cos(pi*x[s]))/D(2) );
        SIDI( 2, D(2), x[s]=x[s]-sin(D(2)*pi*x[s])/(D(2)*pi) );
        SIDI( 3, D(3)/D(4)*pi, x[s]=(D(2)+cos(pi*x[s]))*detail::IPow<D,U,4>::value(sin(pi*x[s]/D(2))) );
        SIDI( 4, D(8)/D(3), x[s]=(D(12)*pi*x[s]-D(8)*sin(D(2)*pi*x[s])+sin(D(4)*pi*x[s]))/(D(12)*pi) );
        SIDI( 5, D(15)/D(16)*pi, x[s]=((D(19)+D(18)*cos(pi*x[s])+D(3)*cos(2*pi*x[s]))*detail::IPow<D,U,6>::value(sin(pi*x[s]/D(2))))/D(4) );
        SIDI( 6, D(16)/D(5), x[s]=-(-D(60)*pi*x[s]+D(45)*sin(D(2)*pi*x[s])-D(9)*sin(D(4)*pi*x[s])+sin(D(6)*pi*x[s]))/(D(60)*pi) );
        SIDI( 7, D(35)/D(32)*pi, x[s]=((D(104)+D(131)*cos(pi*x[s])+D(40)*cos(D(2)*pi*x[s])+D(5)*cos(3*pi*x[s]))*detail::IPow<D,U,8>::value(sin(pi*x[s]/D(2))))/8 );
        SIDI( 8, D(128)/D(35), x[s]=(D(840)*pi*x[s]-D(672)*sin(D(2)*pi*x[s])+D(168)*sin(D(4)*pi*x[s])-D(32)*sin(D(6)*pi*x[s])+D(3)*sin(D(8)*pi*x[s]))/(D(840)*pi) );
        SIDI( 9, D(315)/D(256)*pi, x[s]=((D(2509)+D(3650)*cos(pi*x[s])+D(1520)*cos(D(2)*pi*x[s])+D(350)*cos(D(3)*pi*x[s])+D(35)*cos(D(4)*pi*x[s]))*detail::IPow<D,U,10>::value(sin(pi*x[s]/D(2))))/64 );
        SIDI(10, D(256)/D(63), x[s]=(D(30)*(D(84)*pi*x[s]-D(70)*sin(D(2)*pi*x[s])+D(20)*sin(D(4)*pi*x[s])-D(5)*sin(D(6)*pi*x[s]))+D(25)*sin(D(8)*pi*x[s])-D(2)*sin(D(10)*pi*x[s]))/(2520*pi) );

        #undef SIDI
        #undef HOSTDEVICE

    };
};

#endif
