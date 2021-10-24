#pragma once

#include <array>
#include <utility>

namespace ami::utility {

  template <std::size_t N>
  inline constexpr std::array<std::size_t, N> indices =
    []<std::size_t... I>(std::index_sequence<I...>) {
      return std::array{I...};
    }(std::make_index_sequence<N>{});
}
