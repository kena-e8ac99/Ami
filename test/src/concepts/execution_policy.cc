#include "ami/concepts/execution_policy.hpp"

#include <execution>
#include <tuple>
#include <type_traits>

#include <boost/ut.hpp>

int main() {
  using namespace boost::ut;
  using namespace std::execution;

  "execution_policy"_test = []<class Policy> {
    static_assert(ami::execution_policy<Policy>);
  } | std::tuple{seq, par, par_unseq, unseq};

  "sequenced_policy"_test = []<class Policy> {
    static_assert(ami::sequenced_policy<Policy{}> ==
        std::same_as<std::remove_cvref_t<Policy>, sequenced_policy>);
  } | std::tuple{seq, par, par_unseq, unseq};
}
