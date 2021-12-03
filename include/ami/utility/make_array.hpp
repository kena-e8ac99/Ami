#pragma once

#include <array>
#include <concepts>
#include <random>
#include <type_traits>
#include <utility>

namespace ami::utility {

  template <std::size_t N, std::invocable F>
  inline constexpr std::array<std::invoke_result_t<F>, N> make_array(F f) {
    return [f]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array{(static_cast<void>(I), f())...};
    }(std::make_index_sequence<N>{});
  }

  template <std::size_t N, class F, typename... Args>
  requires (sizeof...(Args) > 0)
  inline constexpr std::array<std::invoke_result_t<F, Args...>, N> make_array(
      F f, Args&&... args) {
    return [f, &args...]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array{(static_cast<void>(I), f(args...))...};
    }(std::make_index_sequence<N>{});
  }

  template <class T, std::size_t N, typename... Args>
  requires std::constructible_from<T, Args...>
  inline constexpr std::array<T, N> make_array(Args&&... args) {
    return [&args...]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array{(static_cast<void>(I), T(args...))...};
    }(std::make_index_sequence<N>{});
  }

  template <std::size_t N, std::floating_point T,
            std::uniform_random_bit_generator G>
  inline constexpr std::array<T, N> make_normal_distributed_array(
      T mean, T stddev, G& engine) {
    std::normal_distribution<T> dist{mean, stddev};

    return make_array<N>([&]() { return dist(engine); });
  }

  template <std::size_t N, std::uniform_random_bit_generator G>
  inline constexpr std::array<bool, N> make_bernouli_distributed_array(
      double p, G& engine) {
    std::bernoulli_distribution dist{p};
    return make_array<N>([&]() { return dist(engine); });
  }
}
