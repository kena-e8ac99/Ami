#include "ami/utility/atomic_operation.hpp"

#include <concepts>
#include <execution>
#include <tuple>

#include <boost/ut.hpp>

int main() {
  using namespace boost::ut;
  using namespace ami::utility;
  using namespace std::execution;

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  "fetch_add"_test = [&]<std::floating_point RealType> {
    should("same result on each execution policy") = [&]<class Policy> {
      RealType value{};
      fetch_add<Policy{}>(value, RealType{1});
      expect(eq(value, RealType{1}));
    } | policies;
  } | std::tuple<float, double>{};
}
