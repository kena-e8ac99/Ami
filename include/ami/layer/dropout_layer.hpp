#pragma once

#include <array>
#include <concepts>
#include <cstddef>

#include "ami/concepts/execution_policy.hpp"

namespace ami {

  template <double DropoutRate>
  requires (DropoutRate > 0.0 && DropoutRate < 1.0)
  struct dropout_layer final {
    template <std::floating_point RealType, std::size_t Size>
    struct type final {
      // Public Types
      using size_type = std::size_t;
      using real_type = RealType;
      using input_type = std::array<real_type, Size>;
      using forward_type = input_type;
      using backward_type = input_type;
      using delta_type = input_type;

      // Public Static Members
      static constexpr size_type input_size = Size;
      static constexpr size_type output_size = Size;
      static constexpr double dropout_rate = DropoutRate;
    };
  };

  template <std::floating_point RealType, std::size_t Size, double DropoutRate>
  using dropout_layer_t =
      typename dropout_layer<DropoutRate>::template type<RealType, Size>;
}
