#pragma once

#include "ami/gate.hpp"

namespace ami {

  template <std::floating_point T, std::size_t N>
  using perceptron = gate<T, N>;
}
