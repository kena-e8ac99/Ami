#pragma once

#include <atomic>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ami/concepts/optimizer.hpp"
#include "ami/utility/atomic_operaton.hpp"
#include "ami/node.hpp"

namespace ami {

  template <std::floating_point T, optimizer<T> O,
            std::size_t N, std::size_t... Ns>
  class gate final {
  public:
    // Static Members
    static constexpr std::size_t node_count = 1 + sizeof...(Ns);

    // Public Types
    using size_type = std::size_t;
    using real_type = T;

    using value_type = std::conditional_t<
      node_count == 1, node<N, T>, std::tuple<node<N, T>, node<Ns, T>...>>;

    using forward_type = T;
    using delta_type   = T;

    template <bool HasInputN = true, bool HasInputNs = true>
    requires (((node_count == 1) && HasInputN) ||
              ((node_count > 1) && (HasInputN || HasInputNs)))
    using backward_type =
      std::conditional_t<
        node_count == 1 || !HasInputNs,
        std::array<T, N>,
        std::conditional_t<
          HasInputN,
          std::tuple<std::array<T, N>, std::array<T, Ns>...>,
          std::conditional_t<
            node_count == 2,
            std::tuple_element_t<node_count - 1,
              std::tuple<std::array<T, N>, std::array<T, Ns>...>>,
            std::tuple<std::array<T, Ns>...>>>>;

    template <bool HasInputN = true, bool HasInputNs = true>
    using gradient_type = std::pair<T, backward_type<HasInputN, HasInputNs>>;

    using optimizer_type = std::conditional_t<
      sizeof...(Ns) == 0, std::pair<O, std::array<O, N>>,
      std::pair<O, std::tuple<std::array<O, N>, std::array<O, Ns>...>>>;

    // Private Types
  private:
    template <bool HasInputN, bool HasInputNs>
    requires (((node_count == 1) && HasInputN) ||
              ((node_count > 1) && (HasInputN || HasInputNs)))
    using input_type = std::conditional_t<
      node_count == 1 || !HasInputNs,
      std::span<const T, N>,
      std::conditional_t<
        HasInputN,
        const std::tuple<std::span<const T, N>, std::span<const T, Ns>...>&,
        const std::tuple<std::span<const T, Ns>...>&>>;

  public:
    // Constructor
    gate() = default;

    explicit constexpr gate(const value_type& value, T bias)
      : node_{value}, bias_{bias} {}

    // Static Methods
    template <execution_policy auto P = std::execution::seq,
              bool HasInputN = true, bool HasInputNs = true>
    requires HasInputN
    static constexpr void calc_gradients(
        input_type<HasInputN, HasInputNs> input, real_type delta,
        gradient_type<HasInputN, HasInputNs>& result) {
      utility::fetch_add(result.first, delta);

      if constexpr (node_count == 1 || !HasInputNs) {
        node<N, T>::template calc_gradients<P>(
            input, delta, std::span{result.second});
      } else {
        [=, &result = result.second]
        <std::size_t... I>(std::index_sequence<I...>) {
          ((std::tuple_element_t<I, value_type>::template calc_gradients<P>(
              std::get<I>(input), delta, std::span{std::get<I>(result)})), ...);
        }(std::make_index_sequence<node_count>{});
      }
    }

    // Public Methods
    template <execution_policy auto P = std::execution::seq,
              bool HasInputN = true, bool HasInputNs = true>
    requires HasInputN
    constexpr real_type forward(input_type<HasInputN, HasInputNs> input) const {
      if constexpr (node_count == 1 || !HasInputNs) {
        return get_node().template forward<P>(input) + bias_;
      } else {
        return
          [&]<std::size_t... I>(std::index_sequence<I...>) {
            return ((get_node<I>().template forward<P>(std::get<I>(input)))
                    + ... + bias_);
          }(std::make_index_sequence<node_count>{});
      }
    }

    template <execution_policy auto P = std::execution::seq,
              bool HasInputN = true, bool HasInputNs = true>
    constexpr void backward(
        T delta, backward_type<HasInputN, HasInputNs>& result) const {
      if constexpr (node_count == 1 || !HasInputNs) {
        get_node().template backward<P>(delta, result);
      } else if constexpr (node_count == 2 && !HasInputN) {
        get_node<1>().template backward<P>(delta, result);
      } else {
        constexpr auto i = static_cast<std::size_t>(!HasInputN);

        [&, delta]<std::size_t... I>(std::index_sequence<I...>) {
          ((get_node<I + i>().template backward<P>(
              delta, std::span{std::get<I>(result)})), ...);
        }(std::make_index_sequence<node_count - i>{});
      }
    }

    template <execution_policy auto P = std::execution::seq, bool X = true>
    requires X
    constexpr void update(optimizer_type&            optimizers,
                          const gradient_type<X, X>& gradients) {
      optimizers.first(bias_, gradients.first);

      if constexpr (node_count == 1) {
        get_node().template update<P>(
            std::span{optimizers.second}, std::span{gradients.second});
      }
      else {
        [&optimizers = optimizers.second, &gradients = gradients.second, this]
        <std::size_t... I>(std::index_sequence<I...>) {
          ((get_node<I>().template update<P>(
              std::span{std::get<I>(optimizers)},
              std::span{std::get<I>(gradients)})), ...);
        }(std::make_index_sequence<node_count>{});
      }
    }

    // Getter / Setter
    template <size_type I = 0>
    requires (I < node_count)
    constexpr auto& get_node() noexcept {
      if constexpr (node_count == 1) {
        return node_;
      }
      else {
        return std::get<I>(node_);
      }
    }

    template <size_type I = 0>
    requires (I < node_count)
    constexpr const auto& get_node() const noexcept {
      if constexpr (node_count == 1) {
        return node_;
      }
      else {
        return std::get<I>(node_);
      }
    }
  private:
    // Private Members
    value_type node_{};

    real_type  bias_{};

    // Private Methods
  };
}
