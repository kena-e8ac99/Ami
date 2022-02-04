#pragma once

#include <concepts>
#include <type_traits>

namespace ami {

  template <class T>
  concept activation_function =
      std::same_as<std::invoke_result_t<
          decltype(&T::template f<float>), float>, float> &&
      std::same_as<std::invoke_result_t<
          decltype(&T::template f<double>), double>, double> &&
      std::same_as<std::invoke_result_t<
          decltype(&T::template df<float>), float>, float> &&
      std::same_as<std::invoke_result_t<
          decltype(&T::template df<double>), double>, double>;
}
