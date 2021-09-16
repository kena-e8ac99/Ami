#include "ami/optimizer/adam.hpp"

#include <algorithm>
#include <array>
#include <functional>

consteval void test_adam() {
  ami::adam adam{};
  std::array<float, 3> gradients{0.1f, 0.2f, 0.3f};
  float weight = 0.1f;

  std::ranges::for_each(gradients, std::bind_front(adam, weight));
}

int main() {
  test_adam();
}
