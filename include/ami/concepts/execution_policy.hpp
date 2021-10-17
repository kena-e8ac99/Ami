#pragma once

#include <concepts>
#include <execution>

namespace ami {

  template <class T>
  concept execution_policy = std::is_execution_policy_v<T>;

  template <auto X>
  concept sequenced_policy
  = std::common_reference_with<decltype(X), std::execution::sequenced_policy>;
}
