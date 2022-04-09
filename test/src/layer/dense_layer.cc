#include "ami/layer/dense_layer.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/ut.hpp>

struct func final {
  template <std::floating_point RealType>
  static constexpr RealType f(RealType value) {
    return value * value;
  }

  template <std::floating_point RealType>
  static constexpr RealType df(RealType value) {
    return static_cast<RealType>(2) * value;
  }
};

template <std::floating_point RealType, std::size_t InputSize,
          std::size_t OutputSize>
consteval void type_check() {
  using layer_t = ami::dense_layer_t<RealType, InputSize, OutputSize, func>;
  static_assert(std::same_as<typename layer_t::size_type, std::size_t>);
  static_assert(std::same_as<typename layer_t::real_type, RealType>);
  static_assert(std::same_as<typename layer_t::value_type,
      std::pair<std::array<std::array<RealType, InputSize>, OutputSize>,
                std::array<RealType, OutputSize>>>);
}

template <class Layer>
void check_value(
    Layer&& layer,
    const typename std::remove_cvref_t<Layer>::value_type& value) {
  using namespace boost::ut;
  using layer_t = std::remove_cvref_t<Layer>;
  auto output = std::forward<Layer>(layer).value();

  std::ranges::for_each(
      std::views::iota(std::size_t{}, layer_t::output_size),
      [&](auto i) {
        expect(std::ranges::equal(output.first[i], value.first[i]));
      });

  expect(std::ranges::equal(output.second, value.second));
}

int main() {
  using namespace boost::ut;
  using namespace ami;
  using namespace std::execution;

  using target_t = std::tuple<
      dense_layer_t<float, 1, 1, func>,  dense_layer_t<float, 1, 2, func>,
      dense_layer_t<float, 2, 1, func>,  dense_layer_t<float, 2, 2, func>,
      dense_layer_t<double, 1, 1, func>, dense_layer_t<double, 1, 2, func>,
      dense_layer_t<double, 2, 1, func>, dense_layer_t<double, 2, 2, func>>;

  "type check"_test = []<std::floating_point RealType>{
    type_check<RealType, 1, 1>();
    type_check<RealType, 1, 2>();
    type_check<RealType, 2, 1>();
    type_check<RealType, 2, 2>();
  } | std::tuple<float, double>{};

  "Getter"_test = []<class Layer> {
    using layer_t = std::remove_cvref_t<Layer>;
    using value_t = typename layer_t::value_type;
    layer_t layer{};
    value_t value{};

    check_value(layer, value);

    [=, layer = layer] {
      check_value(layer, value);
    }();

    check_value(std::move(layer), value);
  } | target_t{};

  "construcotr"_test = []<class Layer> {
    using layer_t = std::remove_cvref_t<Layer>;
    using value_t = typename layer_t::value_type;

    [] {
      Layer layer{value_t{}};
      check_value(layer, value_t{});
    }();
  } | target_t{};

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  //TODO: Implement test
  "forward"_test = [&]<class Layer>(Layer&& layer) {
    using layer_t = std::remove_cvref_t<Layer>;
    typename layer_t::input_type input{};
    should("same result on each execution policy") = [&]<class Policy> {
      [[maybe_unused]]auto result = layer.template forward<Policy{}>(input);
    } | policies;
  } | target_t{};
}
