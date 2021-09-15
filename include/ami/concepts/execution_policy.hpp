#pragma once

#include <execution>

namespace ami {

  template <class T>
  concept execution_policy = std::is_execution_policy_v<T>;
}
