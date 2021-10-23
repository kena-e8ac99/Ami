#pragma once

#include <array>
#include <span>

#include "ami/concepts/activation_function.hpp"
#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  template <std::size_t N, class F, std::floating_point T = float>
  requires activation_function<F, T>
  struct activation_layer final {

    using size_type = std::size_t;

    using real_type = T;

    using input_type = std::span<const T, N>;

    using forward_type = std::array<T, N>;

    using backward_type = std::array<T, N>;

    using delta_type = forward_type;

    struct gradient_type {};

    template <class O>
    struct optimizer_type {};

    template <execution_policy auto P = std::execution::seq>
    static constexpr forward_type forward(input_type input) {
      forward_type output{};
      utility::transform<P>(input, output.begin(), F::template f<T>);
      return output;
    }

    template <execution_policy auto P = std::execution::seq>
    static constexpr backward_type backward(std::span<const T, N> delta,
                                            input_type            input) {
      std::array<T, N> output{};
      utility::transform<P>(
          delta, input, output.begin(),
          [](auto delta, auto input) { return delta * F::df(input); });
      return output;
    }

    template <execution_policy auto P = std::execution::seq, class O>
    constexpr void update(optimizer_type<O>, gradient_type) const noexcept {}
  };
}
