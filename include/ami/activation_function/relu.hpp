#pragma once

#include <concepts>

namespace ami {

  struct relu final {
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
