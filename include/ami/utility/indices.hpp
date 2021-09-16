#pragma once

#include <ranges>
#include <type_traits>

namespace ami::utility {

  template <std::weakly_incrementable auto N>
  inline constexpr auto indices =
    std::ranges::iota_view{std::remove_cvref_t<decltype(N)>{}, N};
}
