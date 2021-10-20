#pragma once

#include <array>

#include "ami/node.hpp"
#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/indices.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  template <std::size_t N, std::size_t M, std::floating_point T = float>
  class fully_connected_layer final {
  public:
    // Public Types
    using size_type = std::size_t;

    using node_type = node<N, T>;

    using real_type = T;

    using forward_type = std::array<real_type, M>;

    using backward_type = typename node_type::backward_type;

    using delta_type = forward_type;

    using gradient_type =
      std::pair<std::array<typename node_type::gradient_type, M>,
                std::array<real_type, M>>;

    template <optimizer<T> O>
    using optimizer_type =
      std::pair<typename node_type::template optimizer_type<O>,
                std::array<O, M>>;

    using input_type = std::span<const real_type, N>;

    // Static Members
    static constexpr size_type output_size = M;

    static constexpr size_type input_size = N;

    // Constructor
    fully_connected_layer() = default;

    explicit fully_connected_layer(
        std::span<const node_type, output_size> values,
        std::span<const real_type, output_size> bias) {
      std::ranges::copy(values, values_.begin());
      std::ranges::copy(bias, bias_.begin());
    }

    // Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradients(input_type            input,
                                         std::span<const T, M> delta,
                                         gradient_type&        gradient) {
      utility::for_each<P>(
          utility::indices<M>,
          [&, input, delta](auto i) {
            utility::fetch_add<P>(gradient.second[i], delta[i]);
            node_type::template calc_gradients<P>(
                input, delta[i], gradient.first[i]);
          });
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr forward_type forward(input_type input) const {
      forward_type output{};
      utility::transform<P>(
          values_, bias_, output.begin(),
          [=](const auto& value, auto bias) {
            return value.template forward<P>(input) + bias;
          });
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
            optimizers.second[i](bias_[i], gradients.second[i]);
            values_[i].template update<P, O>(
                optimizers.first[i], gradients.first[i]); });
    }

    // Getter / Setter
  private:
    // Private Members
    std::array<node_type, output_size> values_{};

    std::array<real_type, output_size>  bias_{};

    // Private Methods
  };
}
