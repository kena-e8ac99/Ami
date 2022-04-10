#include "ami/layer/dropout_layer.hpp"

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

  "type_check"_test = []<class RealType>() {
    type_check<RealType, 1>();
    type_check<RealType, 2>();
  } | std::pair<float, double>{};
}
