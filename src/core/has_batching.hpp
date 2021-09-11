#ifndef QMC_HASBATCHING_H
#define QMC_HASBATCHING_H

#include <utility> // declval
#include <type_traits> // true_type, false_type

namespace integrators
{
    namespace core
    {
        template <typename I, typename T, typename D, typename U, typename = void>
        struct has_batching_impl : std::false_type {};
        template <typename I, typename T, typename D, typename U>
        struct has_batching_impl<I,T,D,U,std::void_t<decltype(std::declval<I>().operator()(std::declval<D*>(),std::declval<T*>(),std::declval<U>()))>> : std::true_type {};

        // Helper function for detecting if the user's functor has a operator(D* x, T* r, U batchsize) used for computing batches of points on CPU
        template <typename I, typename T, typename D, typename U> inline constexpr bool has_batching = has_batching_impl<I,T,D,U>::value;

    };
};

#endif
