subdir('utility')
subdir('optimizer')

test('node_test',
     executable('node_test', 'node.cc',
                dependencies: parallel_test_dep))

test('fully_connected_layer_test',
     executable('fully_connected_layer_test', 'fully_connected_layer.cc',
                dependencies: parallel_test_dep))

test('network_test',
     executable('network_test', 'network.cc',
                dependencies: parallel_test_dep))
