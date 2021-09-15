#include "ami/utility/parallel_algorithm.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <execution>
#include <functional>

#include <boost/ut.hpp>

int main() {
  using namespace boost::ut;

  constexpr auto length = 3;

  constexpr std::array<int, length> src0{1,2,3};

  constexpr std::array<int, length> src1{11,12,13};

  "for_each"_test =
    [&]() {
      constexpr auto my_sum =
        [&](){
          int output = 0;
          ami::utility::for_each(src0, [&](auto i){ output += i; });
          return output;
        }();

      constexpr auto std_sum =
        [&](){
          int output = 0;
          std::ranges::for_each(src0, [&](auto i){ output += i; });
          return output;
        }();

      expect(eq(my_sum, std_sum));
    };

  "parallel_for_each"_test =
    [&]() {
      auto my_sum =
        [&]() {
          int output = 0;
          ami::utility::for_each<std::execution::par>(
              src0,
              [output = std::atomic_ref(output)](auto i) {
                output.fetch_add(i); });
          return output;
        }();

      auto std_sum =
        [&]() {
          int output = 0;
          std::for_each(
              std::execution::par, src0.cbegin(), src0.cend(),
              [output = std::atomic_ref(output)](auto i) {
                output.fetch_add(i); });
          return output;
        }();

      expect(eq(my_sum, std_sum));
    };

  "unary_transform"_test =
    [&]() {
      constexpr auto my_doubles =
        [&]() {
          std::array<int, length> output{};
          ami::utility::transform(src0, output.begin(),
                                  std::bind_front(std::multiplies{}, 2));
          return output;
        }();

      constexpr auto std_doubles =
        [&]() {
          std::array<int, length> output{};
          std::ranges::transform(src0, output.begin(),
                                 std::bind_front(std::multiplies{}, 2));
          return output;
        }();

      for (auto i = 0; i < length; ++i) {
        expect(eq(my_doubles[i], std_doubles[i]));
      }
    };

  "parallel_unary_transform"_test =
    [&]() {
      auto my_doubles =
        [&]() {
          std::array<int, length> output{};
          ami::utility::transform<std::execution::par_unseq>(
              src0, output.begin(), std::bind_front(std::multiplies{}, 2));
          return output;
        }();

      auto std_doubles =
        [&]() {
          std::array<int, length> output{};
          std::transform(std::execution::par_unseq, src0.cbegin(), src0.cend(),
                         output.begin(), std::bind_front(std::multiplies{}, 2));
          return output;
        }();

      for (auto i = 0; i < length; ++i) {
        expect(eq(my_doubles[i], std_doubles[i]));
      }
    };

  "binary_transform"_test =
    [&]() {
      constexpr auto my_sums =
        [&]() {
          std::array<int, length> output{};
          ami::utility::transform(src0, src1, output.begin(), std::plus{});
          return output;
        }();

      constexpr auto std_sums =
        [&]() {
          std::array<int, length> output{};
          std::ranges::transform(src0, src1, output.begin(), std::plus{});
          return output;
        }();

      for (auto i = 0; i < length; ++i) {
        expect(eq(my_sums[i], std_sums[i]));
      }
    };

  "parallel_binary_transform"_test =
    [&]() {
      auto my_sums =
        [&]() {
          std::array<int, length> output{};
          ami::utility::transform<std::execution::unseq>(
              src0, src1, output.begin(), std::plus{});
          return output;
        }();

      auto std_sums =
        [&]() {
          std::array<int, length> output{};
          std::transform(src0.cbegin(), src0.cend(), src1.cbegin(),
                         output.begin(), std::plus{});
          return output;
        }();

      for (auto i = 0; i < length; ++i) {
        expect(eq(my_sums[i], std_sums[i]));
      }
    };

  "binary_transform_reduce"_test =
    [&]() {
      constexpr auto my_sum = ami::utility::transform_reduce(src0, src1, int{});

      constexpr auto std_sum = std::transform_reduce(
          src0.cbegin(), src0.cend(), src1.cbegin(), int{});

      expect(eq(my_sum, std_sum));
    };

  "parallel_binary_transform_reduce"_test =
    [&]() {
      auto my_sum = ami::utility::transform_reduce<std::execution::par>(
          src0, src1, int{});

      auto std_sum = std::transform_reduce(
          std::execution::par, src0.cbegin(), src0.cend(), src1.cbegin(),
          int{});

      expect(eq(my_sum, std_sum));
    };
}
