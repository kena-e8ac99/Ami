#include "ami/network.hpp"

#include <array>
#include <random>

#include <ami/activation_function/sigmoid.hpp>
#include "ami/fully_connected_layer.hpp"
#include "ami/loss_function/mse.hpp"
#include "ami/optimizer/adam.hpp"
#include "ami/optimizer/momentum.hpp"

#include <boost/ut.hpp>

int main() {
  using namespace std::execution;
  using namespace boost::ut;

  std::mt19937 engine{std::random_device{}()};

  ami::network<
    ami::fully_connected_layer<3, 10, double, ami::relu, 0.5>,
    ami::fully_connected_layer<10, 1, double, ami::sigmoid>> src{engine};

  constexpr std::array<std::array<double, 3>, 4> inputs {
    std::array{1.0, 1.0, 1.0}, std::array{0.0, 0.0, 0.0},
    std::array{1.0, 1.0, 0.0}, std::array{1.0, 0.0, 1.0}
  };

  constexpr std::array<std::array<double, 1>, 4> teachers {
    std::array{0.0}, std::array{0.0}, std::array{1.0}, std::array{1.0}
  };

  src.train<ami::mse, ami::adam<double>, 1000, par_unseq>(
      std::span{inputs}, std::span{teachers},
      [](auto x, auto i) {
        if (i % 100 == 0) {
          std::cout << "iteraton " << i << " : accuracy = " << x << '\n';
        }
      });

  std::cout << "output\n";

  for (std::size_t i{}; i != inputs.size(); ++i) {
    auto result = src.forward(inputs[i]);
    std::cout << '[';
    std::ranges::for_each(
        inputs[i],[](auto x) { std::cout << x; });
    std::cout << "] -> [";
    std::ranges::for_each(
        result, [](auto x){ std::cout << x; });
    std::cout << "]\n";
  }
}
