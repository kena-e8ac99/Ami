#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <functional>
#include <ranges>
#include <span>

#include "ami/concepts/execution_policy.hpp"
#include "ami/concepts/optimizer.hpp"
#include "ami/utility/indices.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {

  template <std::size_t N, std::floating_point T = float>
  requires std::atomic_ref<T>::is_always_lock_free
  class node final {
  public:
    // Public Types
    using size_type = std::size_t;
    using real_type = T;

    using value_type = std::array<real_type, N>;

    // Static Members
    static constexpr size_type size = N;

    // Constructor
    node() = default;

    explicit constexpr node(std::span<const real_type, size> value) {
      std::ranges::copy(value, value_.begin());
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr real_type forward(std::span<const real_type, size> input) const {
      return utility::transform_reduce<P>(value_, input, real_type{});
    }

    template <execution_policy auto P = std::execution::seq>
    constexpr void backward(real_type                  delta,
                            std::span<real_type, size> result) const {
      if constexpr (std::common_reference_with<
                      decltype(P), std::execution::sequenced_policy>) {
        std::ranges::transform(
            value_, result, result.begin(),
            [=](auto value, auto result) { return result + (value * delta); });
      }
      else {
        utility::for_each<P>(
            utility::indices<size>,
            [=, this](auto i) {
            std::atomic_ref<real_type>{result[i]}.fetch_add(value_[i] * delta);
            });
      }
    }

    template <execution_policy auto P = std::execution::seq,
              optimizer<real_type>  O>
    constexpr void update(std::span<O, size>               optimizers,
                          std::span<const real_type, size> gradients) {
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
