subdir('utility')
subdir('optimizer')

test('node_test',
     executable('node_test', 'node.cc',
                dependencies: parallel_test_dep))

test('perceptron_test',
     executable('perceptron_test', 'perceptron.cc',
                dependencies: parallel_test_dep))

test('gate_double_test',
     executable('gate_double_test', 'gate_double.cc',
                dependencies: parallel_test_dep))

test('fully_connected_layer_test',
     executable('fully_connected_layer_test', 'fully_connected_layer.cc',
                dependencies: parallel_test_dep))
