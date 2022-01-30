#pragma once

#include <concepts>

namespace ami {

  template <class T>
  concept optimizer =
      std::invocable<T, float&, float> || std::invocable<T, double&, double>;
}
