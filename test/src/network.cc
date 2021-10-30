#include "ami/network.hpp"

#include <array>

#include "ami/fully_connected_layer.hpp"
#include "ami/activation_layer.hpp"
#include "ami/activation_function/relu.hpp"
#include "ami/loss_function/mse.hpp"
#include "ami/optimizer/adam.hpp"

#include <boost/ut.hpp>

int main() {
  using namespace std::execution;
  using namespace boost::ut;

  ami::network<ami::fully_connected_layer<10, 20>,
                         ami::activation_layer<20, ami::relu>,
                         ami::fully_connected_layer<20, 30>> src{};

  constexpr std::array<float, 10> input{};

  const auto output = src.forward(input);

  const auto output1 = src.forward<par_unseq>(input);

  constexpr std::array<float, 30> teacher{};

  src.train<ami::mse, ami::adam<>, seq>(input, teacher);
}
