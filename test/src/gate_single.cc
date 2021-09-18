#include "ami/gate.hpp"

#include <boost/ut.hpp>

struct test_optimizer final {
  constexpr void operator()(float& weight, float value) const {
    weight += value;
  }
};

int main() {
  using namespace boost::ut;

  constexpr auto policy = std::execution::par_unseq;

  constexpr std::size_t size = 3;

  constexpr ami::gate<float, test_optimizer, size>
    src{ami::node<size>{std::array<float, size>{1.0f, 0.1f, 0.01f}}, 1.0f};

  constexpr std::array<float, size> input{0.1f, 0.2f, 0.3f};

  constexpr float delta = 0.1f;

  "forward"_test =
    [&]() {
      constexpr auto output = src.forward(input);

      expect(eq(output, 1.123_f));
    };

  "parallel_forward"_test =
    [&]() {
      auto output = src.forward<policy>(input);

      expect(eq(output, 1.123_f));
    };

  "backward"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          std::array<float, size> output{};
          src.backward(delta, output);
          return output;
        }();

      expect(eq(output[0], 0.1_f));
      expect(eq(output[1], 0.01_f));
      expect(eq(output[2], 0.001_f));
    };

  "parallel_backward"_test =
    [&]() {
      auto output =
        [&]() {
          std::array<float, size> output{};
          src.backward<policy>(delta, output);
          return output;
        }();

      expect(eq(output[0], 0.1_f));
      expect(eq(output[1], 0.01_f));
      expect(eq(output[2], 0.001_f));
    };

  "calc_gradients"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          std::pair<float, std::array<float, size>> output{};
          src.calc_gradients(input, delta, output);
          return output;
        }();

      expect(eq(output.first, 0.1_f));
      expect(eq(output.second[0], 0.01_f));
      expect(eq(output.second[1], 0.02_f));
      expect(eq(output.second[2], 0.03_f));
    };

  "parallel_calc_gradients"_test =
    [&]() {
      auto output =
        [&]() {
          std::pair<float, std::array<float, size>> output{};
          src.calc_gradients<policy>(input, delta, output);
          return output;
        }();

      expect(eq(output.first, 0.1_f));
      expect(eq(output.second[0], 0.01_f));
      expect(eq(output.second[1], 0.02_f));
      expect(eq(output.second[2], 0.03_f));
    };

  std::pair<test_optimizer, std::array<test_optimizer, size>> optimizers{};

  "update"_test =
    [&, src = ami::gate(src)]() mutable {
      src.update(optimizers, std::pair{0.1f, input});

      const auto output = src.get_node().value();

      expect(eq(output[0], 1.1_f));
      expect(eq(output[1], 0.3_f));
      expect(eq(output[2], 0.31_f));
    };

  "parallel_update"_test =
    [&, src = ami::gate(src)]() mutable {
      src.update<policy>(optimizers, std::pair{0.1f, input});

      const auto output = src.get_node().value();

      expect(eq(output[0], 1.1_f));
      expect(eq(output[1], 0.3_f));
      expect(eq(output[2], 0.31_f));
    };
}
