project('Ami', 'cpp',
        default_options:['warning_level=3', 'cpp_std=c++20',
                         'cpp_link_args=-ltbb'])

ami_dep = declare_dependency(
            include_directories: include_directories('include'))

subdir('test')
