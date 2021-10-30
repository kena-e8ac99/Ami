#pragma once

#include "ami/concepts/abstract_layer.hpp"

namespace ami {
  template <class T>
  concept layer
  = abstract_layer<T> ||
    requires (T t) {
      typename T::input_type;
      typename T::forward_type;
      typename T::backward_type;
      typename T::delta_type;
      typename T::gradient_type;

      {T::input_size}  -> std::common_reference_with<typename T::size_type>;
      {T::output_size} -> std::common_reference_with<typename T::size_type>;
    };
}
