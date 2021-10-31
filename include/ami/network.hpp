#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <future>
#include <random>
#include <span>
#include <tuple>
#include <utility>

#include "ami/concepts/layer.hpp"
#include "ami/concepts/execution_policy.hpp"
#include "ami/concepts/optimizer.hpp"
#include "ami/concepts/valid_network.hpp"
#include "ami/utility/atomic_operaton.hpp"
#include "ami/utility/indices.hpp"
#include "ami/utility/parallel_algorithm.hpp"
#include "ami/utility/to_span.hpp"

namespace ami {

  template <layer First, layer... Args>
  requires valid_network<First, Args...>
  class network final {
  public:
    // Public Types
    using size_type = std::size_t;

    using real_type = typename First::real_type;

    using values_type = std::tuple<First, Args...>;

    template <size_type I = 0>
    using value_type = std::tuple_element_t<I, values_type>;

    template <size_type I = 0>
    using input_type = typename value_type<I>::input_type;

    using forwards_type
    = std::tuple<typename First::forward_type, typename Args::forward_type...>;

    template <size_type I = 0>
    using forward_type = typename value_type<I>::forward_type;

    using teacher_type = forward_type<sizeof...(Args)>;

    using backwards_type = std::tuple<typename Args::backward_type...>;

    template <size_type I = 0>
    using backward_type = typename value_type<I>::backward_type;

    template <size_type I = 0>
    using delta_type = typename value_type<I>::delta_type;

    using gradients_type = std::tuple<
      typename First::gradient_type, typename Args::gradient_type...>;

    template <size_type I = 0>
    using gradient_type = typename value_type<I>::gradient_type;

    template <optimizer<real_type> O>
    using optimizers_type
    = std::tuple<typename First::template optimizer_type<O>,
                 typename Args::template optimizer_type<O>...>;

    template <optimizer<real_type> O, size_type I = 0>
    using optimizer_type = typename value_type<I>::template optimizer_type<O>;

    // Static Members
    static constexpr size_type size = sizeof...(Args) + 1;

    template <size_type I = 0>
    static constexpr size_type input_size = value_type<I>::input_size;

    template <size_type I = (size - 1)>
    static constexpr size_type output_size = value_type<I>::output_size;

    // Constructor

    // Static Methods

    // Public Methods
    template <execution_policy auto P = std::execution::seq, size_type I = 0>
    constexpr auto forward(
        utility::to_const_span_t<input_type<I>> input) const {
      if constexpr (I == (size - 1)) {
        return value<I>().template forward<P>(input);
      } else {
        return forward<P, I + 1>(value<I>().template forward<P>(input));
      }
    }

    template <class L, class O, execution_policy auto P = std::execution::seq>
    constexpr void train(utility::to_const_span_t<input_type<0>> input,
                         utility::to_const_span_t<teacher_type>  teacher) {
      gradients_type gradients{};
      real_type      accurecy{};

      train_<L, P, 0>(input, teacher, gradients, accurecy);

      optimizers_type<O> optimizers{};

      update_<P, O>(optimizers, gradients);
    }

    template <class L, class O, std::size_t E,
              execution_policy auto P = std::execution::seq,
              std::size_t N = std::dynamic_extent>
    constexpr void train(std::span<const input_type<0>, N> inputs,
                         std::span<const teacher_type, N>  teachers) {
      optimizers_type<O> optimizers{};

      for (std::size_t epoch = 0; epoch != E; ++epoch) {
        gradients_type gradients{};
        real_type      accurecy{};

        utility::for_each<P>(
            utility::indices<N>,
            [&, inputs, teachers](auto i) {
              train_<L, P, 0>(inputs[i], teachers[i], gradients, accurecy);
            });

        update_<P, O>(optimizers, gradients);
      }
    }

    template <class L, class O, std::size_t B, std::size_t E,
              std::uniform_random_bit_generator G,
              execution_policy auto P = std::execution::seq,
              std::size_t N = std::dynamic_extent>
    constexpr void train(std::span<const input_type<0>, N> inputs,
                         std::span<const teacher_type, N>  teachers, G&& g) {
      std::array<size_type, B> sample{};

      optimizers_type<O> optimizers{};

      constexpr auto epoch = B * E;

      for (std::size_t n = 0; n != epoch; ++n) {
        gradients_type gradients{};
        real_type      accurecy{};

        std::ranges::sample(utility::indices<N>, sample.begin(), B, g);

        utility::for_each<P>(
            sample,
            [&, inputs, teachers](auto i) {
              train_<L, P, 0>(inputs[i], teachers[i], gradients, accurecy);
            });

        update_<P, O>(optimizers, gradients);
      }
    }

    // Getter / Setter
    template <size_type I = 0>
    requires (I < size)
    constexpr auto& value() & noexcept { return std::get<I>(values_); }

    template <size_type I = 0>
    requires (I < size)
    constexpr const auto& value() const& noexcept {
      return std::get<I>(values_);
    }

    template <size_type I = 0>
    requires (I < size)
    constexpr auto value() && noexcept {
      return std::move(std::get<I>(values_));
    }

  private:
    // Private Members
    values_type values_{};

    // Private Methods
    template <class L, execution_policy auto P = std::execution::seq,
              size_type I = 0>
    constexpr backward_type<I> train_(
        utility::to_const_span_t<input_type<I>> input,
        utility::to_const_span_t<teacher_type>  teacher,
        gradients_type& gradients, real_type& accurecy) const {
      delta_type<I>      delta{};

      if constexpr (auto output = value<I>().template forward<P>(input);
                    I == (size - 1)) {
        utility::fetch_add<P>(
            accurecy,
            L::template f<P>(teacher,
                utility::to_const_span_t<forward_type<size - 1>>{output}));

        delta = L::template df<P>(
            teacher, utility::to_const_span_t<forward_type<size - 1>>{output});
      } else {
        delta = train_<L, P, I + 1>(output, teacher, gradients, accurecy);
      }

      if constexpr (I == 0) {
        value_type<I>::template calc_gradients<P>(
            input, delta, std::get<I>(gradients));
        return {};
      }

      if constexpr (abstract_layer<value_type<I>>) {
        return value_type<I>::template backward<P>(delta, input);
      } else if constexpr (sequenced_policy<P>) {
        value_type<I>::template calc_gradients<P>(
            input, delta, std::get<I>(gradients));

        return value<I>().template backward<P>(delta);
      } else {
        auto f = std::async(
            [&]() { return value<I>().template backward<P>(delta); });

        value_type<I>::template calc_gradients<P>(
            input, delta, std::get<I>(gradients));

        return f.get();
      }
    }

    template <execution_policy auto P = std::execution::seq, class O>
    constexpr void update_(optimizers_type<O>&   optimizers,
                           const gradients_type& gradients) {
      if constexpr (sequenced_policy<P>) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          ([&]() {
            value<I>().template update<P, O>(
                std::get<I>(optimizers), std::get<I>(gradients));
           }(), ...);
        }(std::make_index_sequence<size>{});
      } else {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          std::array<std::future<void>, size> fs{};

          ([&]() {
             fs[I] = std::async([&]() {
                 value<I>().template update<P, O>(
                     std::get<I>(optimizers), std::get<I>(gradients)); });
           }(), ...);

          utility::for_each<P>(fs, [](auto&& f) { f.wait(); });
        }(std::make_index_sequence<size>{});
      }
    }
  };
}
