#pragma once

#include <concepts>

namespace ami {

  template <class T, typename U>
  concept optimizer = std::floating_point<U> && std::invocable<T, U&, U>;
}
