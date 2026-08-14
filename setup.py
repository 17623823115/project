from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    
    name="cuda_kernels",
   
    ext_modules=[
        CUDAExtension(
            name="cuda_kernels",
            sources=["cuda_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"]  
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)