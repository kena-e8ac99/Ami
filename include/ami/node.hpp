#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <random>
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
