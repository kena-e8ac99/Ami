#include "ami/fully_connected_layer.hpp"
#include <array>

int main () {
  using namespace std::execution;

  using layer_t = ami::fully_connected_layer<3, 4, float>;

  constexpr layer_t src{};

  layer_t::derivative_type derivative{};

  constexpr auto output = src.forward(std::array<float, 3>{});

  const auto output1 = src.forward<par_unseq>(std::array<float, 3>{});

  auto output2 = src.forward<par_unseq>(std::array<float, 3>{}, derivative);

  constexpr std::array<float, 4> delta{};

  constexpr auto backward = src.backward(delta, layer_t::derivative_type{});

  const auto backward1 = src.backward<par_unseq>(delta, derivative);
}
