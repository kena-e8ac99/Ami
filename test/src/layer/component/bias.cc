#include "ami/layer/component/bias.hpp"

#include <concepts>
#include <tuple>

#include <boost/ut.hpp>

constexpr auto optimizer = [](auto& x, auto y) {
  x += y;
};

using optimizer_t = std::remove_cvref_t<decltype(optimizer)>;

int main() {
  using namespace boost::ut;
  using namespace ami;

  "type_check"_test = []<class RealType> {
    using bias_t = bias<RealType>;

    static_assert(std::same_as<typename bias_t::real_type, RealType>);
    static_assert(std::same_as<
        typename bias_t::template optimizer_type<optimizer_t>, optimizer_t>);
  } | std::tuple<float, double>{};
}
