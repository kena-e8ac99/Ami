#pragma once

#include <cmath>
#include <concepts>

namespace ami {

  struct sigmoid final {
    template <std::floating_point T>
    static
#ifdef __GNUC__
    constexpr
#endif
    T f(T t) {
      return 1 / (1 + std::exp(-t));
    }

    template <std::floating_point T>
    static
#ifdef __GNUC__
      constexpr
#endif
    T df(T t) {
      const auto tmp = f(t);
      return tmp * (1 - tmp);
    }
  };
}
