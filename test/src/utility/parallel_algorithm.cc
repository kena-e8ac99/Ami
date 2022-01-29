#include "ami/utility/parallel_algorithm.hpp"

#include <array>
#include <execution>
#include <utility>
#include <tuple>

#include <boost/ut.hpp>

int main() {
  using namespace boost::ut;
  using namespace ami::utility;
  using namespace std::execution;

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  constexpr std::pair<std::array<int, 1>, std::array<int, 2>>
      test_src{{-1}, {0, 2}};

  "for_each"_test = [&] {
    should("same result on each execution policy") = [&]<class Policy> {
      [&] {
        int value{};
        for_each<Policy{}>(test_src.first, [&](auto i) { value += i; });
        expect(eq(value, -1));
      }();

      [&] {
        int value{};
        for_each<Policy{}>(test_src.second, [&](auto i) { value += i; });
        expect(eq(value, 2));
      }();
    } | policies;
  };

  "transform_reduce(range1, range2, init)"_test = [&] {
    should("same result on each execution policy") = [&]<class Policy> {
      [&] {
        const auto value =
            transform_reduce<Policy{}>(test_src.first, test_src.first, int{1});
        expect(eq(value, 2));
      }();

      [&] {
        const auto value = transform_reduce<Policy{}>(
            test_src.second, test_src.second, int{0});
        expect(eq(value, 4));
      }();
    } | policies;
  };
}
