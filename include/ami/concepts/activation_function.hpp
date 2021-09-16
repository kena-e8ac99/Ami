#pragma once

#include <concepts>
#include <type_traits>

namespace ami {

  template <class T, typename U>
  concept activation_function =
    std::floating_point<U> &&
    std::same_as<U, std::invoke_result_t<decltype(&T::template f<U>), U>> &&
    std::same_as<U, std::invoke_result_t<decltype(&T::template df<U>), U>>;
}
