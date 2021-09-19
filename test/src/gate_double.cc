#include "ami/gate.hpp"

#include <boost/ut.hpp>

struct test_optimizer final {
  constexpr void operator()(float& weight, float value) const {
    weight += value;
  }
};

int main() {
  using namespace boost::ut;

  using namespace std::execution;

  constexpr std::size_t size0 = 3;

  constexpr std::size_t size1 = 4;

  constexpr ami::gate<float, test_optimizer, size0, size1>
    src{std::tuple{ami::node<size0>{std::array<float, size0>{1.0f, 2.0f, 3.0f}},
                   ami::node<size1>{std::array<float, size1>{0.1f, 0.2f, 0.3f}}}, 1.0f};

  constexpr std::array<float, size0> input0{1.0f, 2.0f, 3.0f};
  constexpr std::array<float, size1> input1{0.1f, 0.2f, 0.3f};

  constexpr float delta = 0.1f;

  "forward_all"_test =
    [&]() {
      constexpr auto output = src.forward(
          {std::span{input0}, std::span{input1}});
      expect(eq(output, 15.14_f));
    };

  "forward_first"_test =
    [&]() {
      constexpr auto output = src.forward<seq, true, false>(std::span{input0});
      expect(eq(output, 15.0_f));
    };

  "parallel_forward_all"_test =
    [&]() {
      auto output = src.forward<par_unseq>(
          {std::span{input0}, std::span{input1}});
      expect(eq(output, 15.14_f));
    };

  "parallel_forward_first"_test =
    [&]() {
      auto output = src.forward<par_unseq, true, false>(std::span{input0});
      expect(eq(output, 15.0_f));
    };

  using type = std::remove_cvref_t<decltype(src)>;

  "backward_all"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          type::backward_type<> output{};
          src.backward(delta, output);
          return output;
        }();

      const auto& [first, second] = output;

      expect(eq(first[0], 0.1_f));
      expect(eq(first[1], 0.2_f));
      expect(eq(first[2], 0.3_f));
      expect(eq(second[0], 0.01_f));
      expect(eq(second[1], 0.02_f));
      expect(eq(second[2], 0.03_f));
    };

  "backward_first"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          type::backward_type<true, false> output{};
          src.backward<seq, true, false>(delta, output);
          return output;
        }();

      expect(eq(output[0], 0.1_f));
      expect(eq(output[1], 0.2_f));
      expect(eq(output[2], 0.3_f));
    };

  "backward_second"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          type::backward_type<false, true> output{};
          src.backward<seq, false, true>(delta, output);
          return output;
        }();

      expect(eq(output[0], 0.01_f));
      expect(eq(output[1], 0.02_f));
      expect(eq(output[2], 0.03_f));
    };

  "parallel_backward_all"_test =
    [&]() {
      const auto output =
        [&]() {
          type::backward_type<> output{};
          src.backward<par_unseq>(delta, output);
          return output;
        }();

      const auto& [first, second] = output;

      expect(eq(first[0], 0.1_f));
      expect(eq(first[1], 0.2_f));
      expect(eq(first[2], 0.3_f));
      expect(eq(second[0], 0.01_f));
      expect(eq(second[1], 0.02_f));
      expect(eq(second[2], 0.03_f));
    };

  "parallel_backward_first"_test =
    [&]() {
      const auto output =
        [&]() {
          type::backward_type<true, false> output{};
          src.backward<par_unseq, true, false>(delta, output);
          return output;
        }();

      expect(eq(output[0], 0.1_f));
      expect(eq(output[1], 0.2_f));
      expect(eq(output[2], 0.3_f));
    };

  "backward_second"_test =
    [&]() {
      const auto output =
        [&]() {
          type::backward_type<false, true> output{};
          src.backward<par_unseq, false, true>(delta, output);
          return output;
        }();

      expect(eq(output[0], 0.01_f));
      expect(eq(output[1], 0.02_f));
      expect(eq(output[2], 0.03_f));
    };

  "calc_gradients_all"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          type::gradient_type<> output{};
          src.calc_gradients(
              {std::span{input0}, std::span{input1}}, delta, output);
          return output;
        }();

      const auto& [bias, others] = output;
      const auto& [first, second] = others;

      expect(eq(bias, delta));
      expect(eq(first[0], 0.1_f));
      expect(eq(first[1], 0.2_f));
      expect(eq(first[2], 0.3_f));
      expect(eq(second[0], 0.01_f));
      expect(eq(second[1], 0.02_f));
      expect(eq(second[2], 0.03_f));
    };

  "calc_gradients_first"_test =
    [&]() {
      constexpr auto output =
        [&]() {
          type::gradient_type<true, false> output{};
          src.calc_gradients<seq, true, false>(
              std::span{input0}, delta, output);
          return output;
        }();

      const auto& [bias, value] = output;

      expect(eq(bias, delta));
      expect(eq(value[0], 0.1_f));
      expect(eq(value[1], 0.2_f));
      expect(eq(value[2], 0.3_f));
    };

  "parallel_calc_gradients_all"_test =
    [&]() {
      const auto output =
        [&]() {
          type::gradient_type<> output{};
          src.calc_gradients<par_unseq>(
              {std::span{input0}, std::span{input1}}, delta, output);
          return output;
        }();

      const auto& [bias, others] = output;
      const auto& [first, second] = others;

      expect(eq(bias, delta));
      expect(eq(first[0], 0.1_f));
      expect(eq(first[1], 0.2_f));
      expect(eq(first[2], 0.3_f));
      expect(eq(second[0], 0.01_f));
      expect(eq(second[1], 0.02_f));
      expect(eq(second[2], 0.03_f));
    };

  "calc_gradients_first"_test =
    [&]() {
      const auto output =
        [&]() {
          type::gradient_type<true, false> output{};
          src.calc_gradients<par_unseq, true, false>(
              std::span{input0}, delta, output);
          return output;
        }();

      const auto& [bias, value] = output;

      expect(eq(bias, delta));
      expect(eq(value[0], 0.1_f));
      expect(eq(value[1], 0.2_f));
      expect(eq(value[2], 0.3_f));
    };

  type::optimizer_type optimizers{};

  "update"_test =
    [&, src = ami::gate(src)]() mutable {
      src.update(optimizers, type::gradient_type<>{});
    };

  "parallel_update"_test =
    [&, src = ami::gate(src)]() mutable {
      src.update<par_unseq>(optimizers, type::gradient_type<>{});
    };
}

