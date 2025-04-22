from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="hiradix_schedule_utils",
    ext_modules=[
        CppExtension(
            name="hiradix_schedule_utils",
            sources=["hiradixcache.cpp"],  # your C++ file
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
