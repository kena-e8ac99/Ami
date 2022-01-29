#include "ami/node.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <execution>
#include <tuple>
#include <type_traits>
#include <utility>

#include "boost/ut.hpp"

template <class Node, std::floating_point auto Value>
constexpr auto make_input() {
  using node_t = std::remove_cvref_t<Node>;
  if constexpr (node_t::size == 1) {
    return typename node_t::real_type{Value};
  } else {
    return std::array{typename node_t::real_type{Value},
                      typename node_t::real_type{Value}};
  }
}

int main() {
  using namespace boost::ut;
  using namespace ami;
  using namespace std::execution;

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

  "Getter"_test = []<std::floating_point RealType> {
    [] {
      node<RealType, 1> target{};

      expect(eq(target.value(), RealType{}));
      [target = target] {
        expect(eq(target.value(), RealType{}));
      }();
      expect(eq(std::move(target).value(), RealType{}));
    }();

    [] {
      node<RealType, 2> target{};
      expect(eq(target.value().front(), RealType{}));
      expect(eq(target.value().back(), RealType{}));

      [target = target] {
        expect(eq(target.value().front(), RealType{}));
        expect(eq(target.value().back(), RealType{}));
      }();

      [value = std::move(target).value()] {
        expect(eq(value.front(), RealType{}));
        expect(eq(value.back(), RealType{}));
      }();
    }();
  } | std::pair<float, double>{};

  using test_targets = std::tuple<node<float, 1>, node<float, 2>,
                                  node<double, 1>, node<double, 2>>;

  "constructor"_test = []<class Node> {
      using real_t  = typename Node::real_type;
      using value_t = typename Node::value_type;

    constexpr auto func = [is_first = true] () mutable {
      if (is_first) {
        is_first = false;
        return real_t{-1};
      } else {
        return real_t{1};
      }
    };

    if constexpr (Node::size == 1) {
      expect(eq(Node{value_t{}}.value(), value_t{}));
      expect(eq(Node{func}.value(), value_t{-1}));
    } else {
      [&] {
        Node target{value_t{}};
        expect(eq(target.value().front(), real_t{}));
        expect(eq(target.value().back(), real_t{}));
      }();

      [func] {
        Node target{func};
        expect(eq(target.value().front(), real_t{-1}));
        expect(eq(target.value().back(), real_t{1}));
      }();
    }
  } | test_targets{};

  constexpr std::tuple policies{seq, par, par_unseq, unseq};

  "forward"_test = [&]<class Node>(Node&& src) {
    should("same result on each execution policy") = [&]<class Policy> {
      const auto value =
          src.template forward<Policy{}>(make_input<Node, 1.0>());
      expect(eq(value, typename std::remove_cvref_t<Node>::real_type{0.5}));
    } | policies ;
  } | std::tuple{node<float, 1>{0.5f}, node<float, 2>{{-0.5f, 1.0f}},
                 node<double, 1>{0.5}, node<double, 2>{{-0.5, 1.0}}};
}
