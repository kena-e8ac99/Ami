#include "ami/layer/component/node.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <execution>
#include <tuple>
#include <type_traits>
#include <vector>
#include <utility>

#include "boost/ut.hpp"

constexpr auto optimizer = [](auto& x, auto y) {
  x += y;
};

using optimizer_t = std::remove_cvref_t<decltype(optimizer)>;

template <std::floating_point RealType, std::size_t I>
consteval void type_check() {
  using node_t = ami::node<RealType, I>;
  static_assert(std::same_as<typename node_t::size_type, std::size_t>);
  static_assert(std::same_as<typename node_t::real_type, RealType>);
  static_assert(std::same_as<
      typename node_t::value_type, std::array<RealType, I>>);
  static_assert(std::same_as<typename node_t::forward_type, RealType>);
  static_assert(std::same_as<
      typename node_t::backward_type, std::array<RealType, I>>);
  static_assert(std::same_as<
      typename node_t::template optimizer_type<optimizer_t>,
      std::array<optimizer_t, node_t::size>>);
}

int main() {
  using namespace boost::ut;
  using namespace ami;
  using namespace std::execution;

  "type check"_test = []<class RealType>() {
    type_check<RealType, 1>();
    type_check<RealType, 2>();
  } | std::pair<float, double>{};

  using test_targets = std::tuple<node<float, 1>, node<float, 2>,
                                  node<double, 1>, node<double, 2>>;

  "Getter"_test = []<class Node> {
      Node target{};
      expect(eq(target.value().front(), typename Node::real_type{}));
      expect(eq(target.value().back(), typename Node::real_type{}));

      [target = target] {
        expect(eq(target.value().front(), typename Node::real_type{}));
        expect(eq(target.value().back(), typename Node::real_type{}));
      }();

      [value = std::move(target).value()] {
        expect(eq(value.front(), typename Node::real_type{}));
        expect(eq(value.back(), typename Node::real_type{}));
    }();
  } | test_targets{};

  "constructor"_test = []<class Node> {
      using node_t = std::remove_cvref_t<Node>;
      using real_t  = typename node_t::real_type;
      using value_t = typename node_t::value_type;

    constexpr auto func = [is_first = true] () mutable {
      if (is_first) {
        is_first = false;
        return real_t{-1};
      } else {
        return real_t{1};
      }
    };

    [&] {
      Node target{value_t{}};
      expect(eq(target.value().front(), real_t{}));
      expect(eq(target.value().back(), real_t{}));
    }();

    [&] {
      std::vector<real_t> input{real_t{-1}};
      if constexpr (node_t::size > 1) {
        input.emplace_back(real_t{1});
      }

      Node target{input};
      expect(eq(target.value().front(), real_t{-1}));
      if constexpr (node_t::size > 1) {
        expect(eq(target.value().back(), real_t{1}));
      }
    }();

    [func] {
      Node target{func};
      expect(eq(target.value().front(), real_t{-1}));
      if constexpr (node_t::size > 1) {
        expect(eq(target.value().back(), real_t{1}));
      }
    }();
  } | test_targets{};

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  "forward"_test = [&]<class Node>(Node&& src) {
    should("same result on each execution policy") = [&]<class Policy> {
      using real_t = typename std::remove_cvref_t<Node>::real_type;
      constexpr std::array input{real_t{1}, real_t{1}};
      [&] {
        const auto value = src.template forward<Policy{}>(input);
        expect(eq(value, real_t{0.5}));
      }();

      [&] {
        const auto view = std::views::transform(
            input, [](auto i) { return i * 2; });
        const auto value = src.template forward<Policy{}>(view);
        expect(eq(value, real_t{1}));
      }();
    } | policies ;
  } | std::tuple{node<float, 1>{{0.5f}}, node<float, 2>{{-0.5f, 1.0f}},
                 node<double, 1>{{0.5}}, node<double, 2>{{-0.5, 1.0}}};

  "backward"_test = [&]<class Node>(Node&& src) {
    should("same result on each execution policy") = [&]<class Policy> {
      using node_t = std::remove_cvref_t<Node>;

      typename node_t::backward_type value{};
      src.template backward<Policy{}>(-1, value);

      expect(eq(value.front(), -1 * src.value().front()));
      expect(eq(value.back(), -1 * src.value().back()));
    } | policies ;
  }| std::tuple{node<float, 1>{{0.5f}}, node<float, 2>{{-0.5f, 1.0f}},
                 node<double, 1>{{0.5}}, node<double, 2>{{-0.5, 1.0}}};

  "calc_gradient"_test = [&]<class Node> {
    should("same result on each execution policy") = [&]<class Policy> {
      using node_t = std::remove_cvref_t<Node>;
      using real_t = typename node_t::real_type;

      typename node_t::value_type value{};
      constexpr std::array input{real_t{1}, real_t{1}};

      [&, value]() mutable {
        node_t::template calc_gradient<Policy{}>(input, real_t{0.5}, value);
        expect(eq(value.front(), real_t{0.5}));
        expect(eq(value.back(), real_t{0.5}));
      }();

      [&] {
        const auto view = std::views::transform(
            input, [](auto i) { return i * 2; });

        node_t::template calc_gradient<Policy{}>(view, real_t{0.5}, value);
        expect(eq(value.front(), real_t{1}));
        expect(eq(value.back(), real_t{1}));
      }();
    } | policies ;
  }| test_targets{};

  "update"_test = [&]<class Node> {
    should("same result on each execution policy") = [&]<class Policy> {
      using node_t = std::remove_cvref_t<Node>;
      using real_t = typename node_t::real_type;

      node_t src{};

      std::array<real_t, node_t::size> gradient{real_t{1}};
      gradient.back() = real_t{1};
      typename node_t::template optimizer_type<optimizer_t> optimizers{};

      src.template update<Policy{}>(optimizers, gradient);

      expect(eq(src.value().front(), real_t{1}));
      expect(eq(src.value().back(), real_t{1}));
    } | policies ;
  }| test_targets{};

}
