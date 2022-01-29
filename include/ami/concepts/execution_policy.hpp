#pragma once

#include <concepts>
#include <execution>
#include <type_traits>

namespace ami {

  template <class T>
  concept execution_policy = std::is_execution_policy_v<T>;

  template <auto X>
  concept sequenced_policy = std::same_as<std::remove_cvref_t<decltype(X)>,
          std::execution::sequenced_policy>;
}
