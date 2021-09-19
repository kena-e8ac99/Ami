#pragma once

#include <array>
#include <span>

#include "ami/concepts/activation_function.hpp"
#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  template <std::floating_point T, activation_function<T> F>
  struct activation_layer final {

    template <execution_policy auto P = std::execution::seq, std::size_t N>
    static constexpr std::array<T, N> forward(std::span<const T, N> input) {
      std::array<T, N> output{};
      utility::transform<P>(input, output.begin(), F::template f<T>);
      return output;
    }

    template <execution_policy auto P = std::execution::seq, std::size_t N>
    static constexpr std::array<T, N> backward(std::span<const T, N> delta,
                                               std::span<const T, N> input) {
      std::array<T, N> output{};
      utility::transform<P>(
          delta, input, output.begin(),
          [](auto delta, auto input) { return delta * F::df(input); });
      return output;
    }
  };
}
