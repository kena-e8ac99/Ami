#pragma once

#include <span>

#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  struct mse final {

    template <execution_policy auto P = std::execution::seq,
              std::floating_point T   = float,
              std::size_t N           = std::dynamic_extent>
    static constexpr T f(std::span<const T, N> teacher,
                         std::span<const T, N> input) {
      return utility::transform_reduce<P>(
          teacher, input, T{}, std::plus{},
          [](auto a, auto b) { return (a - b) * (a - b); }) / 2;
    }

    template <execution_policy auto P = std::execution::seq,
              std::floating_point T   = float,
              std::size_t N           = std::dynamic_extent>
    static constexpr std::array<T, N> df(std::span<const T, N> teacher,
                                         std::span<const T, N> input) {
      std::array<T, N> output{};
      utility::transform<P>(teacher, input, output.begin(), std::minus{});
      return output;
    }
  };
}
