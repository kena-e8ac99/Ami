#pragma once

#include "ami/gate.hpp"

namespace ami {

  template <std::floating_point T, optimizer<T> O, std::size_t N>
  using perceptron = gate<T, O, N>;
}
