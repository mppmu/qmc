#include <type_traits> // integral_constant

namespace integrators
{
    namespace transforms
    {
        namespace detail
        {
            /*
             * Power function: ipow<D,U,i>(d) raises the D d to the U power i
             */
            template<typename D, typename U, U exponent>
#ifdef __CUDACC__
            __host__ __device__
#endif
            inline constexpr D ipow(const D base)
            {
                return (exponent == 0) ? 1 : (exponent % 2 == 0) ? ipow<D,U,exponent/2>(base)*ipow<D,U,exponent/2>(base) : base * ipow<D,U,(exponent-1)/2>(base) * ipow<D,U,(exponent-1)/2>(base);
            }

            /*
             * Binomial Coefficients: Binomial<U,n,k>::value gives the type U binomial coefficient (n k)
             */
            template<typename U, U n, U k, typename = void>
            struct Binomial
            {
                constexpr static U value = (Binomial<U,n-1,k-1>::value + Binomial<U,n-1,k>::value);
            };

            // optimisation
            // k > n -k ? bin(n,n-k) : bin(n,k)

            template<typename U, U n, U k>
            struct Binomial<U, n, k, typename std::enable_if<n < k>::type>
            {
                constexpr static U value = 0;
            };

            template<typename U, U n, U k>
            struct Binomial<U, n, k, typename std::enable_if<k == 0>::type>
            {
                constexpr static U value = 1;
            };

            template<typename U, U n, U k>
            struct Binomial<U, n, k, typename std::enable_if<n == k && k != 0>::type>
            {
                constexpr static U value = 1;
            };

            /*
             * Korobov Coefficients and Transform Terms
             */
            template<typename D, typename U, U k, U a, U b, typename = void>
            struct KorobovCoefficient
            {
                constexpr static D value = D(-1)*(b-k+D(1))/k*(a+k)/(a+k+D(1))*KorobovCoefficient<D,U,k-1,a,b>::value;
            };

            template<typename D, typename U, U k, U a, U b>
            struct KorobovCoefficient<D, U, k, a, b, typename std::enable_if<k == 0>::type>
            {
                constexpr static D value = (D(2)*b+D(1))/(a+D(1))*Binomial<U,2*b,b>::value;
            };

            template<typename D, typename U, U k, U a, U b, typename = void>
            struct KorobovTerm
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                constexpr static D value(const D& x)
                {
                    return KorobovTerm<D,U,k-1,a,b>::value(x)*x+KorobovCoefficient<D,U,b-k,a,b>::value;
                }
            };
            template<typename D, typename U, U k, U a, U b>
            struct KorobovTerm<D, U, k, a, b, typename std::enable_if<k == 0>::type>
            {
#ifdef __CUDACC__
                __host__ __device__
#endif
                constexpr static D value(const D& x)
                {
                    return KorobovCoefficient<D,U,b,a,b>::value;
                }
            };
        };

        /*
         * Korobov Transform: Korobov<D,U,r>(x,weight,dim) takes the weight r Korobov transform of x
         */
        template<typename D, typename U, U r>
        struct Korobov
        {
#ifdef __CUDACC__
            __host__ __device__
#endif
            void operator()(D* x, D& wgt, const U dim) const
            {
                D prefactor = (D(2)*r+D(1))*detail::Binomial<U,2*r,r>::value;
                for(U s = 0; s<dim; s++)
                {
                    wgt *= prefactor*detail::ipow<D,U,r>(x[s])*detail::ipow<D,U,r>(D(1)-x[s]);
                    x[s] = detail::ipow<D,U,r+1>(x[s])*detail::KorobovTerm<D,U,r,r,r>::value(x[s]);
                    // loss of precision can cause x > 1., must keep in x \elem [0,1]
                    if (x[s] > D(1))
                        x[s] = D(1);
                }
            }
        };

        // TODO make 3 a compiler template argument
        template <typename D, typename U>
        struct Korobov3
        {
#ifdef __CUDACC__
            __host__ __device__
#endif
            void operator()(D* x, D& wgt, const U dim) const
            {
                // Korobov r = 3
                for (U s = 0; s < dim; s++)
                {
                    wgt *= x[s] * x[s] * x[s] * D(140)*(D(1) - x[s])*(D(1) - x[s])*(D(1) - x[s]);
                    x[s] = x[s] * x[s] * x[s] * x[s] * (D(35) + x[s] * (D(-84) + x[s] * (D(70) + x[s] * D(-20))));
                    // loss of precision can cause x > 1., must keep in x \elem [0,1]
                    if (x[s] > D(1))
                        x[s] = D(1);
                }
            }
        };

        template<typename D, typename U>
        struct Trivial
        {
#ifdef __CUDACC__
            __host__ __device__
#endif
            void operator()(D* x, D& wgt, const U dim) const {}
        };

    };

};

