#include "ami/network.hpp"

#include <array>
#include <random>

#include "ami/fully_connected_layer.hpp"
#include "ami/loss_function/mse.hpp"
#include "ami/optimizer/adam.hpp"

#include <boost/ut.hpp>

int main() {
  using namespace std::execution;
  using namespace boost::ut;

  ami::network<ami::fully_connected_layer<10, 20>,
               ami::fully_connected_layer<20, 30>> src{};

  constexpr std::array<float, 10> input{};

  const auto output = src.forward(input);

  const auto output1 = src.forward<par_unseq>(input);

  constexpr std::array<float, 30> teacher{};

  src.train<ami::mse, ami::adam<>>(input, teacher);

  src.train<ami::mse, ami::adam<>, par_unseq>(input, teacher);

  constexpr std::array<std::array<float, 10>, 10> inputs{};

  constexpr std::array<std::array<float, 30>, 10> teachers{};

  src.train<ami::mse, ami::adam<>, 10, seq, 10>(inputs, teachers);

  src.train<ami::mse, ami::adam<>, 10, par_unseq, 10>(inputs, teachers);

  std::mt19937 engine{std::random_device{}()};

  src.train<ami::mse, ami::adam<>, 8, 10, std::mt19937, seq, 10>(
      inputs, teachers, engine);

  src.train<ami::mse, ami::adam<>, 8, 10, std::mt19937, par_unseq, 10>(
      inputs, teachers, engine);
}
