#include "ami/layer/activation_layer.hpp"

#include <array>
#include <execution>
#include <tuple>
#include <type_traits>
#include <utility>

#include "boost/ut.hpp"

struct func {
  template <std::floating_point RealType>
  static constexpr RealType f(RealType src) { return src; }

  template <std::floating_point RealType>
  static constexpr RealType df(RealType src) { return src; }
};

template <std::floating_point RealType, std::size_t I>
consteval void type_check() {
  using layer_t = ami::activation_layer_t<RealType, I, func>;
  static_assert(std::same_as<typename layer_t::size_type, std::size_t>);
  static_assert(std::same_as<typename layer_t::real_type, RealType>);
  static_assert(std::same_as<
      typename layer_t::forward_type, std::array<RealType, I>>);
  static_assert(std::same_as<
      typename layer_t::backward_type, std::array<RealType, I>>);
}

int main() {
  using namespace ami;
  using namespace boost::ut;
  using namespace std::execution;

  "type_check"_test = []<class RealType>() {
    type_check<RealType, 1>();
    type_check<RealType, 2>();
  } | std::pair<float, double>{};

  using test_target_t = std::tuple<
      activation_layer_t<float, 1, func>, activation_layer_t<float, 2, func>,
      activation_layer_t<double, 1, func>, activation_layer_t<double, 2, func>>;

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  "forward"_test = [&]<class Layer>() {
    using layer_t = std::remove_cvref_t<Layer>;
    typename layer_t::input_type input{};

    should("same result on each policy") = [&]<class Policy>() {
      [[maybe_unused]] auto result = layer_t::template forward<Policy{}>(input);
    } | policies;
  } | test_target_t{};
}
