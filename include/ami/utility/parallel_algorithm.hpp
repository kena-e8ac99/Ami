#pragma once

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>

#include "ami/concepts/execution_policy.hpp"

namespace ami::utility {

  template <execution_policy auto Policy,
            std::ranges::forward_range R,
            std::indirectly_unary_invocable<std::ranges::iterator_t<R>> F>
  inline constexpr void for_each(R&& r, F f) {
    if constexpr (sequenced_policy<Policy>) {
      std::ranges::for_each(std::forward<R>(r), std::move(f));
    } else {
      std::for_each(
          Policy, std::ranges::begin(r), std::ranges::end(r), std::move(f));
    }
  }

  template <execution_policy auto Policy,
            std::ranges::forward_range R1, std::ranges::forward_range R2,
            std::common_with<std::ranges::range_value_t<R1>> T>
  requires std::common_with<T, std::ranges::range_value_t<R2>>
  inline constexpr T transform_reduce(R1&& r1, R2&& r2, T init) {
    if constexpr (sequenced_policy<Policy>) {
      return std::transform_reduce(
          std::ranges::begin(r1), std::ranges::end(r1), std::ranges::begin(r2),
          std::move(init));
    } else {
      return std::transform_reduce(
          Policy, std::ranges::begin(r1), std::ranges::end(r1),
          std::ranges::begin(r2), std::move(init));
    }
  }
}
