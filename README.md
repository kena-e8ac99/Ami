# Ami [WIP]
C++ 20, Non dependency, header only machine learning library.

Currently this library implements the basic functionality of mlp.
# Example

```cpp
#include "ami/fully_connected_layer.hpp"
#include "ami/network.hpp"

ami::network<ami::fully_connected_layer<2, 3>,
             ami::fully_connected_layer<3, 4>,
             ami::fully_connected_layer<4, 5>> net{};

std::array<float, 2> input{};

auto output = net.forward(input);

output = net.forward(std::execution::par_unseq, input);

std::array<float, 5> teacher{};

net.train(input, teacher);

net.train(std::execution::par_unseq, input, teacher);
```

# Tested Compiler

- GCC   11.1.0
