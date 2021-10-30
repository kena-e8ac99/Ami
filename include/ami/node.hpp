#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <functional>
#include <ranges>
#include <span>

#include "ami/concepts/execution_policy.hpp"
#include "ami/concepts/optimizer.hpp"
#include "ami/utility/atomic_operaton.hpp"
#include "ami/utility/indices.hpp"
#include "ami/utility/parallel_algorithm.hpp"
#include "ami/utility/to_span.hpp"

namespace ami {

  template <std::size_t N, std::floating_point T = float>
  requires std::atomic_ref<T>::is_always_lock_free
  class node final {
  public:
    // Public Types
    using size_type = std::size_t;
    using real_type = T;

    using value_type = std::array<real_type, N>;

    using input_type = value_type;

    using forward_type = real_type;

    using backward_type = value_type;

    using delta_type = forward_type;

    using gradient_type = value_type;

    template <optimizer<real_type> O>
    using optimizer_type = std::array<O, N>;

    // Static Members
    static constexpr size_type size = N;

    // Constructor
    node() = default;

    explicit constexpr node(std::span<const real_type, size> value) {
      std::ranges::copy(value, value_.begin());
    }

    // Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradients(
        utility::to_const_span_t<input_type> input, real_type delta,
        utility::to_span_t<gradient_type>    result) {
      utility::for_each<P>(
          utility::indices<size>,
          [=](auto i) { utility::fetch_add<P>(result[i], input[i] * delta); });
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr real_type forward(
        utility::to_const_span_t<input_type> input) const {
      return utility::transform_reduce<P>(value_, input, real_type{});
    }

    template <execution_policy auto P = std::execution::seq>
    constexpr void backward(delta_type                        delta,
                            utility::to_span_t<backward_type> result) const {
      utility::for_each<P>(
          utility::indices<size>,
          [=, this](auto i) {
            utility::fetch_add<P>(result[i], value_[i] * delta); });
    }

    template <execution_policy auto P = std::execution::seq,
              optimizer<real_type>  O>
    constexpr void update(std::span<O, size>                      optimizers,
                          utility::to_const_span_t<gradient_type> gradients) {
      utility::for_each<P>(
          utility::indices<size>,
          [=, this](auto i) { optimizers[i](value_[i], gradients[i]); });
    }

    // Getter / Setter
    constexpr std::span<real_type, size> value() & noexcept {
      return std::span{value_};
    }

    constexpr std::span<const real_type, size> value() const& noexcept {
      return std::span{value_};
    }

    constexpr value_type value() && noexcept {
      return std::move(value_);
    }
  private:
    // Private Members
    value_type value_{};

    // Private Methods
  };
}
