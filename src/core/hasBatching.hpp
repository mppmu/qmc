#ifndef QMC_HASBATCHING_H
#define QMC_HASBATCHING_H

template <typename T, typename DP, typename RP, typename U> static constexpr bool hasBatching(...) {
    return false;
}

template <typename T, typename DP, typename RP, typename U> static constexpr bool hasBatching(int, decltype(std::is_void<decltype(std::declval<T>().operator()(DP(), RP(), U()))>::value) result = false) {
    return std::is_void<decltype(std::declval<T>().operator()(DP(), RP(), U()))>::value;
}

#endif
