#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <execution>

#include "ami/concepts/activation_function.hpp"
#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  template <activation_function F>
  struct activation_layer final {
    template <std::floating_point RealType, std::size_t Size>
    struct type final {
      // Public Types
      using size_type = std::size_t;
      using real_type = RealType;
      using input_type = std::array<real_type, Size>;
      using forward_type = input_type;
      using backward_type = input_type;

      // Public Static Members
      static constexpr size_type input_size = Size;
      static constexpr size_type output_size = Size;

      // Public Static Methods
      template <execution_policy auto P = std::execution::seq>
      static constexpr forward_type forward(const input_type& input) {
        forward_type result{};
        utility::transform<P>(input, result.begin(), F::template f<real_type>);
        return result;
      }
    };
  };

  template <std::floating_point RealType, std::size_t Size,
            activation_function F>
  using activation_layer_t =
      typename activation_layer<F>::template type<RealType, Size>;
}
