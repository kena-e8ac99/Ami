#pragma once

#include <concepts>

#include "ami/concepts/optimizer.hpp"

namespace ami {

  template <std::floating_point RealType>
  class bias final {
  public:
    // Public Types
    using real_type = RealType;

    template <optimizer Optimizer>
    using optimizer_type = Optimizer;

    // Constructor
    bias() = default;

    explicit constexpr bias(real_type value) noexcept : value_{value} {}

    // Getter
    constexpr real_type value() const noexcept { return value_; }

  private:
    // Private Members
    real_type value_{};
  };
}
