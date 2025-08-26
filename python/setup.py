import sys
import platform
import subprocess
import os
from setuptools import setup, Extension
import pybind11
import numpy

# --- Configuration ---
WOWLIB_CPP_DIR = "../wow"  # Directory containing wow_index.h, utils.hh, fixed_string.h

# Base compile arguments
compile_args = [
    "-std=c++20",  # Explicitly set C++20 standard
    "-O3",
    "-DNDEBUG",
    "-Wall",
    "-Wno-sign-compare",
    "-Wno-narrowing",
    "-Wno-reorder",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-fPIC",
]
link_args = []

# Platform-specific configuration
if sys.platform == "darwin":  # macOS
    # macOS doesn't have MADV_HUGEPAGE
    compile_args.append("-DMADV_HUGEPAGE=0")
    
    # Architecture-specific flags
    machine = platform.machine().lower()
    if machine in ["arm64", "aarch64"]:
        # Apple Silicon - use native optimization
        compile_args.append("-mcpu=native")
    else:
        # Intel Mac - can use x86 SIMD
        compile_args.extend(["-march=native", "-mavx2", "-mavx", "-msse"])
        
    # OpenMP for macOS
    try:
        # Try to find OpenMP paths
        openmp_paths = [
            "/opt/homebrew/opt/libomp",  # Apple Silicon Homebrew
            "/usr/local/opt/libomp",     # Intel Homebrew
            "/opt/local",                # MacPorts
        ]
        
        openmp_found = False
        for path in openmp_paths:
            if os.path.exists(os.path.join(path, "include", "omp.h")):
                compile_args.extend(["-Xpreprocessor", "-fopenmp", f"-I{path}/include"])
                link_args.extend([f"-L{path}/lib", "-lomp"])
                openmp_found = True
                break
                
        if not openmp_found:
            print("Warning: OpenMP not found. Install with: brew install libomp")
            
    except Exception as e:
        print(f"Warning: Could not configure OpenMP: {e}")
        
else:  # Linux and other Unix-like systems
    compile_args.extend(["-lrt", "-fpic", "-march=native"])
    
    # Add OpenMP flags
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")


def check_simd_support(arch):
    """Check if SIMD instruction set is supported on the current platform."""
    if sys.platform == "win32":
        # Windows-specific check
        try:
            result = subprocess.run(["wmic", "cpu", "get", "caption"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    elif sys.platform == "darwin":  # macOS
        try:
            if arch == "avx512f":
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.features"], 
                                      capture_output=True, text=True, timeout=5)
                return "AVX512F" in result.stdout.upper()
            else:
                # For other SIMD, just return True on Intel Macs, False on ARM
                machine = platform.machine().lower()
                return machine not in ["arm64", "aarch64"]
        except:
            return False
    else:  # Linux/Unix
        try:
            with open("/proc/cpuinfo", "r") as f:
                return arch in f.read()
        except:
            return False


# Only add x86 SIMD flags if we're not on ARM64
machine = platform.machine().lower()
if machine not in ["arm64", "aarch64"]:
    if check_simd_support("avx512f"):
        compile_args.append("-mavx512f")
    if check_simd_support("avx2"):
        compile_args.append("-mavx2")
    if check_simd_support("avx"):
        compile_args.append("-mavx")
    if check_simd_support("sse"):
        compile_args.append("-msse")

# Define the extension module
ext_modules = [
    Extension(
        "pywowlib._pywowlib_core",
        ["wow_bindings.cc"],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            WOWLIB_CPP_DIR,
        ],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

# Call setup - now configuration comes from pyproject.toml
setup(
    ext_modules=ext_modules,
    zip_safe=False,
)
