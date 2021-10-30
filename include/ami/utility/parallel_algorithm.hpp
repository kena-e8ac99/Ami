#pragma once

#include <algorithm>
#include <numeric>
#include <ranges>

#include "ami/concepts/execution_policy.hpp"

namespace ami::utility {

  // for_each
  template <execution_policy auto P = std::execution::seq,
            std::ranges::forward_range R,
            std::indirectly_unary_invocable<std::ranges::iterator_t<R>> F>
  inline constexpr void for_each(R&& r, F f) {
    if constexpr (sequenced_policy<P>) {
      std::for_each(std::ranges::begin(r), std::ranges::end(r), std::move(f));
    }
    else {
      std::for_each(
          P, std::ranges::begin(r), std::ranges::end(r), std::move(f));
    }
  }

  // transform
  template <execution_policy auto P = std::execution::seq,
            std::ranges::forward_range R,
            std::weakly_incrementable O, std::copy_constructible F>
  requires std::indirectly_writable<O,
             std::indirect_result_t<F, std::ranges::iterator_t<R>>>
  inline constexpr O transform(R&& r, O result, F f) {
    if constexpr (sequenced_policy<P>) {
      return std::transform(std::ranges::begin(r), std::ranges::end(r),
                            std::move(result), std::move(f));
    }
    else {
      return std::transform(P, std::ranges::begin(r), std::ranges::end(r),
                            std::move(result), std::move(f));
    }
  }

  template <execution_policy auto P = std::execution::seq,
            std::ranges::forward_range R1, std::ranges::forward_range R2,
            std::weakly_incrementable O, std::copy_constructible F>
  requires std::indirectly_writable<O,
             std::indirect_result_t<F, std::ranges::iterator_t<R1>,
               std::ranges::iterator_t<R2>>>
  inline constexpr O transform(R1&& r1, R2&& r2, O result, F f) {
    if constexpr (sequenced_policy<P>) {
      return std::transform(std::ranges::begin(r1), std::ranges::end(r1),
                            std::ranges::begin(r2), std::move(result),
                            std::move(f));
    }
    else {
      return std::transform(P, std::ranges::begin(r1), std::ranges::end(r1),
                            std::ranges::begin(r2), std::move(result),
                            std::move(f));
    }
  }

  // transform_reduce
  template <execution_policy auto P = std::execution::seq,
            std::ranges::forward_range R1, std::ranges::forward_range R2,
            std::common_with<std::ranges::range_value_t<R1>> T>
  requires std::common_with<T, std::ranges::range_value_t<R2>>
  inline constexpr T transform_reduce(R1&& r1, R2&& r2, T init) {
    if constexpr (sequenced_policy<P>) {
      return std::transform_reduce(std::ranges::begin(r1), std::ranges::end(r1),
                                   std::ranges::begin(r2), std::move(init));
    }
    else {
      return std::transform_reduce(
               P, std::ranges::begin(r1), std::ranges::end(r1),
               std::ranges::begin(r2), std::move(init));
    }
  }

  template <execution_policy auto P = std::execution::seq,
            std::ranges::forward_range R1, std::ranges::forward_range R2,
            typename T, std::copy_constructible F1, std::copy_constructible F2>
  requires std::common_with<T,
             std::invoke_result_t<F1, std::ranges::range_value_t<R1>,
                                  std::ranges::range_value_t<R2>>> &&
           std::common_with<T,
             std::invoke_result_t<F2, std::ranges::range_value_t<R1>,
                                  std::ranges::range_value_t<R2>>>
  inline constexpr T transform_reduce(R1&& r1, R2&& r2, T init, F1 f1, F2 f2) {
    if constexpr (sequenced_policy<P>) {
      return
        std::transform_reduce(
            std::ranges::begin(r1), std::ranges::end(r1),
            std::ranges::begin(r2), std::move(init), std::move(f1),
            std::move(f2));
    }
    else {
      return
        std::transform_reduce(
            P, std::ranges::begin(r1), std::ranges::end(r1),
            std::ranges::begin(r2), std::move(init), std::move(f1),
            std::move(f2));
    }
  }
}
