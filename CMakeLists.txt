# cmake minimum required version
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

project(wow)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Platform-specific compiler flags
if(APPLE)
    # Base flags for macOS
    set(CMAKE_CXX_FLAGS "-O3 -Wall -DHAVE_CXX0X -fPIC -w -finline-functions")
    
    # Architecture-specific SIMD flags
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        # Apple Silicon (ARM64) - use NEON instead of x86 SIMD
        message(STATUS "Building for Apple Silicon (ARM64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    else()
        # Intel Mac - use x86 SIMD
        message(STATUS "Building for Intel Mac (x86_64)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mavx -msse")
        
        # Check AVX512F support on Intel Mac
        execute_process(COMMAND sysctl -n machdep.cpu.features OUTPUT_VARIABLE CPU_FEATURES)
        if(CPU_FEATURES MATCHES "AVX512F")
            message(STATUS "AVX512F supported on Intel Mac")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
        else()
            message(STATUS "AVX512F not supported on Intel Mac")
        endif()
    endif()
    
    # Try to find and enable OpenMP on macOS
    # Note: If OpenMP is installed in a non-standard location, set OPENMP_ROOT:
    # export OPENMP_ROOT=/path/to/your/openmp/installation
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        message(STATUS "OpenMP found and enabled")
    else()
        # Try to find OpenMP manually for macOS with various package managers
        find_path(OPENMP_INCLUDE_DIR omp.h
            HINTS
                ENV OPENMP_ROOT
            PATHS 
                /opt/homebrew/opt/libomp/include    # Homebrew on Apple Silicon
                /opt/homebrew/include               # Homebrew fallback
                /usr/local/opt/libomp/include       # Homebrew on Intel
                /usr/local/include                  # Homebrew fallback
                /opt/local/include                  # MacPorts
                /usr/include                        # System
            PATH_SUFFIXES include
        )
        
        find_library(OPENMP_LIBRARY 
            NAMES omp gomp iomp5
            HINTS
                ENV OPENMP_ROOT
            PATHS 
                /opt/homebrew/opt/libomp/lib        # Homebrew on Apple Silicon
                /opt/homebrew/lib                   # Homebrew fallback
                /usr/local/opt/libomp/lib           # Homebrew on Intel
                /usr/local/lib                      # Homebrew fallback
                /opt/local/lib                      # MacPorts
                /usr/lib                            # System
            PATH_SUFFIXES lib
        )
        
        if(OPENMP_INCLUDE_DIR AND OPENMP_LIBRARY)
            get_filename_component(OPENMP_LIB_DIR ${OPENMP_LIBRARY} DIRECTORY)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${OPENMP_INCLUDE_DIR}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${OPENMP_LIB_DIR} -lomp")
            message(STATUS "OpenMP manually found: ${OPENMP_INCLUDE_DIR}, ${OPENMP_LIBRARY}")
        else()
            message(WARNING "OpenMP not found. Install with: brew install libomp")
            message(WARNING "Or set OPENMP_ROOT environment variable to your OpenMP installation")
        endif()
    endif()
else()
    # Linux flags
    set(CMAKE_CXX_FLAGS "-O3 -Wall -march=native -lrt -DHAVE_CXX0X -fpic -w -fopenmp -mavx2 -mavx -msse -finline-functions")
    
    # check if avx512f is supported on Linux
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        execute_process(COMMAND grep -c avx512f /proc/cpuinfo OUTPUT_VARIABLE AVX512F_SUPPORTED)
        if (AVX512F_SUPPORTED GREATER 0)
            message(STATUS "AVX512F supported")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
        else()
            message(STATUS "AVX512F not supported")
        endif()
    endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -g -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0 -g -fPIC")

add_subdirectory(example)