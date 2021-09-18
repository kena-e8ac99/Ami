subdir('utility')
subdir('optimizer')

test('node_test',
     executable('node_test', 'node.cc',
                dependencies: parallel_test_dep))

test('gate_single_test',
     executable('gate_single_test', 'gate_single.cc',
                dependencies: parallel_test_dep))

test('gate_double_test',
     executable('gate_double_test', 'gate_double.cc',
                dependencies: parallel_test_dep))
