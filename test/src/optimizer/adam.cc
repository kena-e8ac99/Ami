#include "ami/optimizer/adam.hpp"

#include <concepts>
#include <tuple>

#include <boost/ut.hpp>

#include "ami/concepts/optimizer.hpp"

int main() {
  using namespace boost::ut;
  using namespace ami;

  "adam"_test = []<std::floating_point RealType> {
    adam_t<RealType> src{};

    static_assert(optimizer<adam_t<RealType>>);

    const auto func = [&] {
      RealType value{};
      src(value, RealType{1});
      return value;
    };

    expect(neq(func(), func()));
  } | std::tuple<float, double>{};
}
