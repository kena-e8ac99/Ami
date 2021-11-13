#pragma once

#include <concepts>

namespace ami {

  template <std::floating_point T = float, T LR = T{0.01}, T M = T{0.9}>
  class momentum final {
  public:
    constexpr void operator()(T& weight, T gradient) {
      prev_ = LR * gradient - M * prev_;

      weight -= prev_;
    }
  private:
    T prev_{};
  };
}
