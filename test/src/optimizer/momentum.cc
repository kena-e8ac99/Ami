#include "ami/optimizer/momentum.hpp"

#include <iostream>

constexpr float grad_func(float x) noexcept {
  return 2 * x - 2;
}

int main() {
  std::size_t i{};

  float weight{};

  ami::momentum<> momentum{};

  for (bool is_converged{false}; !is_converged; ++i) {
    auto dw = grad_func(weight);

    auto tmp = weight;

    momentum(weight, dw);

    is_converged = (weight == tmp);

    std::cout << "iteration " << i << " : " << weight << '\n';
  }

  std::cout << "converged after " << i << " iterations" << '\n';
}
