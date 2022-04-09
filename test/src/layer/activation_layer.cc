#include "ami/layer/activation_layer.hpp"

#include <array>
#include <tuple>
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

  "type_check"_test = []<class RealType>() {
    type_check<RealType, 1>();
    type_check<RealType, 2>();
  } | std::pair<float, double>{};
}
