#pragma once

#include <array>
#include <cmath>
#include <random>

#include "ami/node.hpp"
#include "ami/concepts/execution_policy.hpp"
#include "ami/utility/indices.hpp"
#include "ami/utility/parallel_algorithm.hpp"
#include "ami/utility/to_span.hpp"

namespace ami {

  template <std::size_t N, std::size_t M, std::floating_point T = float>
  class fully_connected_layer final {
  public:
    // Public Types
    using size_type = std::size_t;

    using node_type = node<N, T>;

    using real_type = T;

    using value_type
    = std::pair<std::array<node_type, M>, std::array<real_type, M>>;

    using input_type = std::array<real_type, N>;

    using forward_type = std::array<real_type, M>;

    using backward_type = typename node_type::backward_type;

    using delta_type = forward_type;

    using gradient_type =
      std::pair<std::array<typename node_type::gradient_type, M>,
                std::array<real_type, M>>;

    template <optimizer<T> O>
    using optimizer_type =
      std::pair<std::array<typename node_type::template optimizer_type<O>, M>,
                std::array<O, M>>;

    // Static Members
    static constexpr size_type output_size = M;

    static constexpr size_type input_size = N;

    // Constructor
    fully_connected_layer() = default;

    explicit constexpr fully_connected_layer(const value_type& value)
      : nodes_{value.first}, bias_{value.second} {}

    explicit constexpr fully_connected_layer(value_type&& value)
      : nodes_{std::move(value.first)}, bias_{std::move(value.second)} {}

    template <std::uniform_random_bit_generator G>
    explicit constexpr fully_connected_layer(
        G& engine, real_type mean = 0,
        real_type stddev = std::sqrt(real_type{2 / N}))
      : nodes_{utility::make_array<node_type, M>(mean, stddev, engine)} {}

    // Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradients(
        utility::to_const_span_t<input_type> input,
        utility::to_const_span_t<delta_type> delta,
        gradient_type&                       gradient) {
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
    constexpr forward_type forward(
        utility::to_const_span_t<input_type> input) const {
      forward_type output{};
      utility::transform<P>(
          nodes_, bias_, output.begin(),
          [=](const auto& value, auto bias) {
            return value.template forward<P>(input) + bias;
          });
      return output;
    }

    template <execution_policy auto P = std::execution::seq>
    constexpr backward_type backward(
        utility::to_const_span_t<delta_type> delta) const {
      backward_type output{};
      utility::for_each<P>(
          utility::indices<M>,
          [&, delta](auto i) {
            nodes_[i].template backward<P>(delta[i], output); });
      return output;
    }

    template <execution_policy auto P = std::execution::seq, optimizer<T> O>
    constexpr void update(optimizer_type<O>&   optimizers,
                          const gradient_type& gradients) {
      utility::for_each<P>(
          utility::indices<output_size>,
          [&](auto i) {
            optimizers.second[i](bias_[i], gradients.second[i]);
            nodes_[i].template update<P, O>(
                optimizers.first[i], gradients.first[i]); });
    }

    // Getter / Setter
    constexpr auto value() const& {
      std::array<typename node_type::value_type, output_size> output{};

      std::ranges::transform(
          nodes_, output.begin(), [](auto&& node) { return node.value(); });

      return std::pair{std::move(output), bias_};
    }

    constexpr auto value() && {
      std::array<typename node_type::value_type, output_size> output{};

      std::ranges::transform(
          nodes_, output.begin(),
          [](auto&& node) { return std::move(node).value(); });

      return std::pair{std::move(output), std::move(bias_)};
    }


  private:
    // Private Members
    std::array<node_type, output_size> nodes_{};

    std::array<real_type, output_size>  bias_{};

    // Private Methods
  };
}
