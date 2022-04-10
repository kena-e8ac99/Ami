#include "ami/layer/dropout_layer.hpp"

#include <random>
#include <tuple>
#include <utility>

#include "boost/ut.hpp"

template <std::floating_point RealType, std::size_t I>
consteval void type_check() {
  using layer_t = ami::dropout_layer_t<RealType, I, 0.5>;
  static_assert(std::same_as<typename layer_t::size_type, std::size_t>);
  static_assert(std::same_as<typename layer_t::real_type, RealType>);
  static_assert(std::same_as<
      typename layer_t::forward_type, std::array<RealType, I>>);
  static_assert(std::same_as<
      typename layer_t::backward_type, std::array<RealType, I>>);
  static_assert(std::same_as<
      typename layer_t::delta_type, std::array<RealType, I>>);
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
      dropout_layer_t<float, 1, 0.5>, dropout_layer_t<float, 2, 0.5>,
      dropout_layer_t<double, 1, 0.5>, dropout_layer_t<double, 2, 0.5>>;

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  std::default_random_engine engine{std::random_device{}()};

  "forward"_test = [&]<class Layer>() {
    using layer_t = std::remove_cvref_t<Layer>;
    typename layer_t::input_type input{};

    should("same result on each policy") = [&]<class Policy>() {
      [[maybe_unused]] auto result =
          layer_t::template forward<Policy{}>(input, engine);
    } | policies;
  } | test_target_t{};

  "backward"_test = [&]<class Layer>() {
    using layer_t = std::remove_cvref_t<Layer>;
    typename layer_t::forward_type output{};
    typename layer_t::delta_type delta{};

    should("same result on each policy") = [&]<class Policy>() {
      [[maybe_unused]] auto result =
          layer_t::template backward<Policy{}>(output, delta);
    } | policies;
  } | test_target_t{};

}
