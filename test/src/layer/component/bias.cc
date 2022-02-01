#include "ami/layer/component/bias.hpp"

#include <concepts>
#include <execution>
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

  using target_t = std::tuple<bias<float>, bias<double>>;

  "getter"_test = []<class Bias>(Bias src) {
    expect(eq(src.value(), typename Bias::real_type{}));

    [src = src] {
      expect(eq(src.value(), typename Bias::real_type{}));
    }();

    [value = std::move(src).value()] {
      expect(eq(value, typename Bias::real_type{}));
    }();
  } | target_t{};

  "constructor"_test = []<class Bias> {
    constexpr auto value = typename Bias::real_type{1};
    Bias src{value};

    expect(eq(src.value(), value));
  } | target_t{};

  using namespace std::execution;
  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  "calc_gradient"_test = [&]<class Bias>(Bias src) {
    using real_t = typename Bias::real_type;
    should("same result on each policy") = [&]<class Policy> {
      real_t value{};
      src.template calc_gradient<Policy{}>(real_t{1}, value);
      expect(eq(value, real_t{1}));
    } | policies;
  } | target_t{};
}
