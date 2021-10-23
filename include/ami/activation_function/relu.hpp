#pragma once

#include <concepts>

#include "ami/activation_layer.hpp"

namespace ami {

  struct relu final {
    template <std::size_t N, std::floating_point T = float>
    using layer_type = activation_layer<N, relu, T>;

    template <std::floating_point T>
    static constexpr T f(T t) noexcept {
      return t > 0 ? t : 0;
    }

    template <std::floating_point T>
    static constexpr T df(T t) noexcept {
      return t > 0 ? 1 : 0;
    }
  };
}
