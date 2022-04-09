#include "ami/concepts/activation_function.hpp"

#include <concepts>

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

int main() {
  using namespace boost::ut;
  using namespace ami;

  "activation_function"_test = [] {
    static_assert(activation_function<func>);
  };
}
