#pragma once

#include <tuple>

namespace ami::utility {

  template <class... T>
  struct composited;

  template <typename T>
  struct composited<T> final {
    using type = T;
  };

  template <typename T, typename U>
  struct composited<T, U> final {
    using type = std::pair<T, U>;
  };

  template <typename... T>
  requires (sizeof...(T) > 2)
  struct composited<T...> final {
    using type = std::tuple<T...>;
  };

  template <typename... T>
  using composited_t = typename composited<T...>::type;
}
