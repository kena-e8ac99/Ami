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
  };
}
