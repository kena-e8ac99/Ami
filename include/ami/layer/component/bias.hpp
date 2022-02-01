#pragma once

#include <concepts>

#include "ami/concepts/execution_policy.hpp"
#include "ami/concepts/optimizer.hpp"
#include "ami/utility/atomic_operation.hpp"

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

    // Public Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradient(real_type delta, real_type& result) {
      utility::fetch_add<P>(result, delta);
    }

    // Getter
    constexpr real_type value() const noexcept { return value_; }

  private:
    // Private Members
    real_type value_{};
  };
}
