#pragma once

#include <cmath>
#include <concepts>

namespace ami {

  struct adam final {
    template <std::floating_point RealType,
              RealType LearningRate = RealType{0.001},
              RealType Beta1        = RealType{0.9},
              RealType Beta2        = RealType{0.999},
              RealType Eps          = RealType{1e-7}>
    class type final {
    public:

#ifdef __GNUC__
      constexpr
#endif
        void operator()(RealType& weight, RealType gradient) {
          m_ = m_ * Beta1 + (RealType{1} - Beta1) * gradient;
          v_ = v_ * Beta2 + (RealType{1} - Beta2) * gradient * gradient;

          weight -=
              LearningRate * (m_ / (static_cast<RealType>(1) - pow_beta1_))
              / (std::sqrt(v_ / (static_cast<RealType>(1) - pow_beta2_)) + Eps);

          pow_beta1_ *= Beta1;
          pow_beta2_ *= Beta2;
        }

    private:
      RealType m_{};
      RealType v_{};
      RealType pow_beta1_{Beta1};
      RealType pow_beta2_{Beta2};
    };
  };

    template <std::floating_point RealType,
              RealType LearningRate = RealType{0.001},
              RealType Beta1        = RealType{0.9},
              RealType Beta2        = RealType{0.999},
              RealType Eps          = RealType{1e-7}>
    using adam_t =
        typename adam::type<RealType, LearningRate, Beta1, Beta2, Eps>;
}
