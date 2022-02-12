#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include "ami/concepts/execution_policy.hpp"
#include "ami/concepts/optimizer.hpp"
#include "ami/utility/atomic_operation.hpp"
#include "ami/utility/parallel_algorithm.hpp"

namespace ami {
  template<std::floating_point RealType, std::size_t Size>
  requires (Size > 0)
  class node final {
  public:
    // Public Types
    using size_type     = std::size_t;
    using real_type     = RealType;
    using value_type    = std::array<RealType, Size>;
    using forward_type  = real_type;
    using backward_type = value_type;

    template <optimizer Optimizer>
    using optimizer_type = std::array<Optimizer, Size>;

    // Static Public Members
    static constexpr size_type size = Size;

    // Constructor
    node() = default;

    explicit constexpr node(const value_type& value) noexcept : value_{value} {}

    explicit constexpr node(value_type&& value) noexcept
      : value_{std::move(value)} {}

    template <std::ranges::input_range R>
    requires (!std::same_as<std::remove_cvref_t<R>, value_type>)
    explicit constexpr node(const R& value) {
      std::ranges::copy(value, value_.begin());
    }

    template <std::invocable Func>
    requires std::constructible_from<real_type, std::invoke_result_t<Func>>
    explicit constexpr node(Func func) noexcept(noexcept(func()))
      : value_ { [func]<std::size_t... I>(std::index_sequence<I...>) mutable {
          return value_type{(static_cast<void>(I), func())...};
        }(std::make_index_sequence<size>{}) } {}

    // Public Static Methods
    template <execution_policy auto P = std::execution::seq>
    static constexpr void calc_gradient(
        const std::ranges::forward_range auto& input, real_type delta,
        value_type& result) {
      utility::for_each<P>(std::views::iota(size_type{}, size),
          [&, delta](auto i) {
            utility::fetch_add<P>(result[i], delta * input[i]);
          });
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr real_type forward(
        const std::ranges::forward_range auto& input) const {
      return utility::transform_reduce<P>(value_, input, real_type{});
    }

    template <execution_policy auto P = std::execution::seq>
    constexpr void backward(real_type delta, backward_type& result) const {
      utility::for_each<P>(std::views::iota(size_type{}, size),
          [&, delta](auto i) {
            utility::fetch_add<P>(result[i], delta * value_[i]);
          });
    }

    template <execution_policy auto P = std::execution::seq, class Optimizer>
    constexpr void update(
        optimizer_type<Optimizer>& optimizer, const value_type& gradient) {
      utility::for_each<P>(std::views::iota(size_type{}, size), [&](auto i) {
            optimizer[i](value_[i], gradient[i]);
          });
    }

    // Getter
    constexpr value_type value() const& noexcept {
      return value_;
    }

    constexpr value_type value() && noexcept { return std::move(value_); }

  private:
    // Private Members
    value_type value_{};
  };
}
