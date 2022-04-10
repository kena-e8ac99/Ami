#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <random>
#include <utility>

#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami::detail {
  template <std::size_t N, std::uniform_random_bit_generator G>
  inline std::array<bool, N> make_bernouli_array(double p, G& engine) {
    return [&engine, dist = std::bernoulli_distribution(p)]<std::size_t... I>(
        std::index_sequence<I...>) mutable {
          return std::array{(static_cast<void>(I), dist(engine))...};
        }(std::make_index_sequence<N>{});
  }
}

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

      // Public Static Methods
      template <execution_policy auto P = std::execution::seq,
                std::uniform_random_bit_generator G>
      static forward_type forward(const input_type& input, G& engine) {
        forward_type result{};
        utility::transform<P>(
            input,
            detail::make_bernouli_array<output_size>(dropout_rate, engine),
            result.begin(), [](auto&& input, auto p) {
              return p ? input / dropout_rate : real_type{0};
            });
        return result;
      }

      template <execution_policy auto P = std::execution::seq>
      static backward_type backward(
          const forward_type& forward, const delta_type& delta) {
        backward_type result{};
        utility::transform<P>(
            forward, delta, result.begin(), [](auto&& forward, auto&& delta) {
              return (forward == real_type{0}) ? forward : delta;
            });
        return result;
      }
    };
  };

  template <std::floating_point RealType, std::size_t Size, double DropoutRate>
  using dropout_layer_t =
      typename dropout_layer<DropoutRate>::template type<RealType, Size>;
}
