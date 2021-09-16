#include "ami/utility/indices.hpp"

#include <boost/ut.hpp>

int main() {
  using namespace boost::ut;

  using length_type = std::size_t;

  constexpr length_type length = 10;

  constexpr auto my_indices = ami::utility::indices<length>;

  constexpr auto std_indices = std::ranges::iota_view{length_type{}, length};

  for (length_type i = 0; i < length; ++i) {
    expect(eq(my_indices[i], std_indices[i]));
  }
}
