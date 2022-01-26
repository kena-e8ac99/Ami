#include "ami/node.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <utility>

#include "boost/ut.hpp"

int main() {
  using namespace boost::ut;
  using namespace ami;

  "type check"_test = []<class RealType>() {
    static_assert(std::same_as<
        typename node<RealType, 1>::size_type, std::size_t>);

    static_assert(std::same_as<
        typename node<RealType, 1>::real_type, RealType>);

    static_assert(std::same_as<
        typename node<RealType, 1>::value_type, RealType>);
    static_assert(std::same_as<
        typename node<RealType, 2>::value_type, std::array<RealType, 2>>);

    static_assert(std::same_as<
        typename node<RealType, 1>::forward_type, RealType>);

    static_assert(std::same_as<
        typename node<RealType, 1>::backward_type, RealType>);
    static_assert(std::same_as<typename
        node<RealType, 2>::backward_type, std::array<RealType, 2>>);
  } | std::pair<float, double>{};
}
