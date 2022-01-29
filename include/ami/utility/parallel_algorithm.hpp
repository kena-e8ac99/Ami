#pragma once

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>

#include "ami/concepts/execution_policy.hpp"

namespace ami::utility {

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
