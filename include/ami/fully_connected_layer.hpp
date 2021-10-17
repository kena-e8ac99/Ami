#pragma once

#include <array>

#include "ami/concepts/execution_policy.hpp"
#include "ami/perceptron.hpp"
#include "ami/utility/indices.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  template <std::size_t N, std::size_t M, std::floating_point T = float>
  class fully_connected_layer final {
  public:
    // Public Types
    using size_type = std::size_t;

    using value_type = perceptron<N, T>;

    using real_type = typename value_type::real_type;

    using forward_type = std::array<real_type, M>;

    using backward_type = typename value_type::backward_type;

    using gradient_type =
      std::array<typename value_type::gradient_type, M>;

    template <optimizer<T> O>
    using optimizer_type =
      std::array<typename value_type::template optimizer_type<O>, M>;

    using input_type = std::span<const real_type, N>;

    // Static Members
    static constexpr size_type output_size = M;

    static constexpr size_type input_size = N;

    // Constructor
    fully_connected_layer() = default;

    explicit fully_connected_layer(
        std::span<const value_type, output_size> values) {
      std::ranges::copy(values, values_.begin());
    }

    // Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradients(input_type            input,
                                         std::span<const T, M> delta,
                                         gradient_type&        gradient) {
      utility::for_each<P>(
          utility::indices<M>,
          [&, input, delta](auto i) {
            value_type::template calc_gradients<P>(
                input, delta[i], gradient[i]);
          });
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr forward_type forward(input_type input) const {
      forward_type output{};
      utility::transform<P>(
          values_, output.begin(),
          [=](const auto& value) { return value.template forward<P>(input); });
      return output;
    }

    template <execution_policy auto P = std::execution::seq>
    constexpr backward_type backward(std::span<const T, M> delta) const {
      backward_type output{};
      utility::for_each<P>(
          utility::indices<M>,
          [&, delta](auto i) {
            values_[i].template backward<P>(delta[i], output); });
      return output;
    }

    template <execution_policy auto P = std::execution::seq, optimizer<T> O>
    constexpr void update(optimizer_type<O>&   optimizers,
                          const gradient_type& gradients) {
      utility::for_each<P>(
          utility::indices<output_size>,
          [&](auto i) {
            values_[i].template update<P, O>(optimizers[i], gradients[i]); });
    }

    // Getter / Setter
  private:
    // Private Members
    std::array<value_type, output_size> values_{};

    // Private Methods
  };
}
