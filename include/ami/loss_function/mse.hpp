#pragma once

#include <span>

#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  struct mse final {

    template <execution_policy auto P = std::execution::seq,
              std::floating_point T   = float,
              std::size_t N           = std::dynamic_extent>
    static constexpr T f(std::span<const T, N> output,
                         std::span<const T, N> teacher) {
      return utility::transform_reduce<P>(
          output, teacher, T{}, std::plus{},
          [](auto a, auto b) { return (a - b) * (a - b); }) / N;
    }

    template <execution_policy auto P = std::execution::seq,
              std::floating_point T   = float,
              std::size_t N           = std::dynamic_extent>
    static constexpr std::array<T, N> df(std::span<const T, N> output,
                                         std::span<const T, N> teacher) {
      std::array<T, N> result{};
      utility::transform<P>(
          output, teacher, result.begin(),
          [](auto a, auto b) { return (a - b) * 2 / N; });
      return result;
    }
  };
}
