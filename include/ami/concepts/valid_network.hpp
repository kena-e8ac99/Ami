#pragma once

#include <type_traits>

#include "ami/concepts/layer.hpp"

namespace ami {

  template <class T, class... Args>
  struct is_valid_network : std::false_type {};

  template <layer T, layer U>
  requires (T::output_size == U::input_size)
  struct is_valid_network<T, U> : std::true_type {};

  template <layer T, layer U, layer... Args>
  requires (T::output_size == U::input_size)
  struct is_valid_network<T, U, Args...> : is_valid_network<U, Args...> {};

  template <class T, class... Args>
  inline constexpr bool is_valid_network_v
  = is_valid_network<T, Args...>::value;

  template <class T, class... Args>
  concept valid_network = is_valid_network_v<T, Args...>;
}
