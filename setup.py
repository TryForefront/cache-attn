from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

torch.utils.cpp_extension.COMMON_NVCC_FLAGS = [
    "--expt-relaxed-constexpr"
]  # This is necessary to enable half precision conversions

setup(
    name="cache_attn",
    ext_modules=[
        CUDAExtension(
            "cache_attn",
            ["extension.cpp", "kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
