#pragma once

#include <atomic>

#include "ami/concepts/execution_policy.hpp"

namespace ami::utility {

  template <execution_policy auto P = std::execution::seq, typename T>
  requires std::atomic_ref<T>::is_always_lock_free
  inline constexpr T fetch_add(
      T& t,
      typename std::atomic_ref<T>::difference_type operand,
      [[maybe_unused]] std::memory_order order = std::memory_order_seq_cst) {
    if constexpr (sequenced_policy<P>) {
      T temp = t;
      t += operand;
      return temp;
    }
    else {
      return std::atomic_ref<T>{t}.fetch_add(operand, order);
    }
  }
}
