#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "ami/concepts/execution_policy.hpp"
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

    // Static Public Members
    static constexpr size_type size = Size;

    // Constructor
    node() = default;

    explicit node(const value_type& value) noexcept : value_{value} {}

    explicit node(value_type&& value) noexcept : value_{value} {}

    template <std::ranges::input_range R>
    requires (!std::same_as<std::remove_cvref_t<R>, value_type>)
    explicit constexpr node(const R& value)
      : value_{[&]{
          value_type result{};
          std::ranges::copy(value, std::ranges::begin(result));
          return result;
        }()} {}

    template <std::invocable Func>
    requires std::constructible_from<real_type, std::invoke_result_t<Func>>
    explicit node(Func func) noexcept(noexcept(func()))
      : value_ { [func]<std::size_t... I>(std::index_sequence<I...>) mutable {
          return value_type{(static_cast<void>(I), func())...};
        }(std::make_index_sequence<size>{}) } {}

    // Public Methods
    template <execution_policy auto P = std::execution::seq>
    constexpr real_type forward(
        const std::ranges::forward_range auto& input) const {
      return utility::transform_reduce<P>(value_, input, real_type{});
    }

    // Getter
    constexpr const value_type& value() const& noexcept { return value_; }

    constexpr value_type value() && noexcept { return std::move(value_); }

  private:
    // Private Members
    value_type value_{};
  };
}
