#pragma once

#include <concepts>

#include "ami/node.hpp"

namespace ami {

  template <std::size_t N, std::floating_point T = float>
  class perceptron final {
  public:
    // Public Types
    using size_type = std::size_t;

    using real_type = T;

    using node_type = node<N, T>;

    using value_type = std::pair<T, node_type>;

    using input_type = std::span<const T, N>;

    using forward_type = T;

    using delta_type = T;

    using backward_type = std::array<T, N>;

    using gradient_type = std::pair<T, std::array<T, N>>;

    template <optimizer<T> O>
    using optimizer_type = std::pair<O, std::array<O, N>>;

    // Static Members
    static constexpr size_type size = N;

    // Constructor
    perceptron() = default;

    explicit constexpr perceptron(const value_type& value)
      : bias_{value.first}, node_{value.second} {}

    // Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradients(input_type input, delta_type delta,
                                         gradient_type& result) {
      utility::template fetch_add<P>(result.first, delta);
      node_type::template calc_gradients<P>(input, delta, result.second);
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr forward_type forward(input_type input) const {
      return node_.template forward<P>(input) + bias_;
    }

    template <execution_policy auto P = std::execution::seq>
    constexpr void backward(delta_type delta, backward_type& result) const {
      node_.template backward<P>(delta, result);
    }

    template <execution_policy auto P = std::execution::seq, optimizer<T> O>
    constexpr void update(optimizer_type<O>& optimizer,
                          const gradient_type& gradient) {
      optimizer.first(bias_, gradient.first);
      node_.template update<P>(
          std::span{optimizer.second}, std::span{gradient.second});
    }

    // Getter / Setter
    constexpr node_type& get_node() & noexcept { return node_; }

    constexpr const node_type& get_node() const& noexcept { return node_; }

    constexpr node_type get_node() && noexcept { return std::move(node_); }

    constexpr value_type& get_value() & noexcept { return {node_, bias_}; }

    constexpr const value_type& get_value() const& noexcept {
      return {node_, bias_};
    }

    constexpr value_type get_value() && noexcept {
      return {std::move(node_), std::move(bias_)};
    }

  private:
    // Private Members
    real_type bias_{};

    node_type node_{};
    // Private Methods
  };
}
