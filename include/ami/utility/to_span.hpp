#pragma once

#include <array>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>

namespace ami::utility {

  // to_span
  template <class T>
  struct to_span;

  template <typename T, std::size_t N>
  struct to_span<std::array<T, N>> final {
    using type = std::span<T, N>;
  };

  template <typename T, std::size_t N, std::size_t M>
  struct to_span<std::pair<std::array<T, N>, std::array<T, M>>> final {
    using type = std::pair<std::span<T, N>, std::span<T, M>>;
  };

  template <typename T, std::size_t... Ns>
  struct to_span<std::tuple<std::array<T, Ns>...>> final {
    using type = std::tuple<std::span<T, Ns>...>;
  };

  template <class T>
  using to_span_t = typename to_span<T>::type;

  // to_const_span
  template <class T>
  struct to_const_span;

  template <typename T, std::size_t N>
  struct to_const_span<std::array<T, N>> final {
    using type = std::span<const T, N>;
  };

  template <typename T, std::size_t N, std::size_t M>
  struct to_const_span<std::pair<std::array<T, N>, std::array<T, M>>> final {
    using type = std::pair<std::span<const T, N>, std::span<const T, M>>;
  };

  template <typename T, std::size_t... Ns>
  struct to_const_span<std::tuple<std::array<T, Ns>...>> final {
    using type = std::tuple<std::span<const T, Ns>...>;
  };

  template <class T>
  using to_const_span_t = typename to_const_span<T>::type;
}
