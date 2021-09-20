#include "ami/fully_connected_layer.hpp"
#include "ami/perceptron.hpp"
#include <array>

int main () {
  using namespace std::execution;

  constexpr ami::fully_connected_layer<ami::perceptron<float, 3>, 4> src{};

  constexpr auto output = src.forward(std::array<float, 3>{});

  const auto output1 = src.forward<par_unseq>(std::array<float, 3>{});

  constexpr std::array<float, 4> delta{};

  constexpr auto backward = src.backward(delta);

  const auto backward1 = src.backward<par_unseq>(delta);
}
