from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="group_requests_ext",
    ext_modules=[
        CppExtension(
            name="group_requests_ext",
            sources=["group_requests.cpp"],  # your C++ file
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
