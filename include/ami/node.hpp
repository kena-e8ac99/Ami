#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ami {
  template<std::floating_point RealType, std::size_t Size>
  requires (Size > 0)
  class node final {
  public:
    // Public Types
    using size_type     = std::size_t;
    using real_type     = RealType;
    using value_type    =
        std::conditional_t<Size == 1, real_type, std::array<RealType, Size>>;
    using forward_type  = real_type;
    using backward_type = value_type;

    // Static Public Members
    static constexpr size_type size = Size;

    // Constructor
    node() = default;

    explicit node(value_type value) noexcept requires (size == 1)
      : value_{value} {}

    explicit node(const value_type& value) noexcept requires (size > 1)
      : value_{value} {}

    explicit node(value_type&& value) noexcept requires (size > 1)
      : value_{value} {}

    template <std::invocable Func>
    requires std::constructible_from<real_type, std::invoke_result_t<Func>> &&
             (size == 1)
    explicit node(Func func) noexcept(noexcept(func()))
      : value_ { func() } {}

    template <std::invocable Func>
    requires std::constructible_from<real_type, std::invoke_result_t<Func>> &&
             (size > 1)
    explicit node(Func func) noexcept(noexcept(func()))
      : value_ { [func]<std::size_t... I>(std::index_sequence<I...>) mutable {
          return value_type{(static_cast<void>(I), func())...};
        }(std::make_index_sequence<size>{}) } {}

    // Getter
    constexpr value_type value() const& noexcept requires (size == 1) {
      return value_;
    }

    constexpr const value_type& value() const& noexcept requires (size == 2) {
      return value_;
    }

    constexpr value_type value() && noexcept requires (size == 2) {
      return std::move(value_);
    }

  private:
    // Private Members
    value_type value_{};
  };
}
