#include "ami/node.hpp"

#include <boost/ut.hpp>

int main() {
  using namespace boost::ut;

  constexpr std::size_t size = 3;

  constexpr ami::node<size> src{std::array<float, size>{1.0f, 0.1f, 0.01f}};

  constexpr std::array<float, size> input{0.1f, 0.2f, 0.3f};

  constexpr float delta = 0.1f;

  "forward"_test =
    [&]() {
      constexpr auto output = src.forward(input);

      expect(eq(output, 0.123_f));
    };

  "parallel_forward"_test =
    [&]() {
      const auto output = src.forward<std::execution::par_unseq>(input);

      expect(eq(output, 0.123_f));
    };

  "backward"_test =
    [&]() {
      std::array<float, size> output{};
      src.backward(delta, output);

      expect(eq(output[0], 0.1_f));
      expect(eq(output[1], 0.01_f));
      expect(eq(output[2], 0.001_f));
    };

  "parallel_backward"_test =
    [&]() {
      std::array<float, size> output{};
      src.backward<std::execution::par_unseq>(delta, output);

      expect(eq(output[0], 0.1_f));
      expect(eq(output[1], 0.01_f));
      expect(eq(output[2], 0.001_f));
    };

  "calc_update"_test =
    [&]() {
      std::array<float, size> output{};
      src.calc_update(input, delta, output);

      expect(eq(output[0], 0.01_f));
      expect(eq(output[1], 0.02_f));
      expect(eq(output[2], 0.03_f));
    };

  "parallel_calc_update"_test =
    [&]() {
      std::array<float, size> output{};
      src.calc_update<std::execution::par_unseq>(input, delta, output);

      expect(eq(output[0], 0.01_f));
      expect(eq(output[1], 0.02_f));
      expect(eq(output[2], 0.03_f));
    };

  constexpr auto test_optimizer =
    [](auto& weight, auto gradient) { weight += gradient; };

  std::array optimizers{test_optimizer, test_optimizer, test_optimizer};

  "update"_test =
    [&, src = ami::node(src)]() mutable {
      src.update(std::span{optimizers}, std::span{input});

      const auto output = src.value();

      expect(eq(output[0], 1.1_f));
      expect(eq(output[1], 0.3_f));
      expect(eq(output[2], 0.31_f));
    };

  "parallel_update"_test =
    [&, src = ami::node(src)]() mutable {
      src.update<std::execution::par_unseq>(std::span{optimizers},
                                            std::span{input});
      const auto output = src.value();

      expect(eq(output[0], 1.1_f));
      expect(eq(output[1], 0.3_f));
      expect(eq(output[2], 0.31_f));
    };
}
