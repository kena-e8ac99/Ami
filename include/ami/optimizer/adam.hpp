#pragma once

#include <cmath>
#include <concepts>

namespace ami {

  template <std::floating_point T = float, T LR= T{0.001}, T Beta1 = T{0.9},
            T Beta2 = T{0.999}, T Eps = T{1e-8}>
  class adam final {
  public:

#ifdef __GNUC__
    constexpr
#endif
    void operator()(T& weight, T gradient) {
      m_ = (m_ ? Beta1 * m_ : T{}) + (T{1} - Beta1) * gradient;
      v_ = (v_ ? Beta2 * v_ : T{}) + (T{1} - Beta2) * gradient * gradient;

      weight -= LR * (m_ / (T{1} - pow_beta1))
                / (std::sqrt(v_ / (T{1} - pow_beta2)) + Eps);

      pow_beta1 *= Beta1;
      pow_beta2 *= Beta2;
    }

  private:
    T m_{};
    T v_{};

    T pow_beta1{Beta1};
    T pow_beta2{Beta2};
  };
}
