#pragma once

#include <concepts>
#include <type_traits>

#include "ami/activation_layer.hpp"

namespace ami {

  template <class T>
  struct is_abstract_layer final : public std::false_type {};

  template <std::floating_point T, activation_function<T> F>
  struct is_abstract_layer<activation_layer<T, F>> final
  : public std::true_type {};

  template <class T>
  inline constexpr bool is_abstract_layer_v = is_abstract_layer<T>::value;

  template <class T>
  concept abstract_layer = is_abstract_layer_v<T>;
}
