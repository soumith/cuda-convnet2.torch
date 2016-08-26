CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 7.5 REQUIRED)

IF(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
# New CMake function, comes with FindCUDA
  CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
ELSE()
  SET(NVCC_FLAGS_EXTRA "-gencode=arch=compute_30,code=sm_30;-gencode=arch=compute_35,code=sm_35;-gencode=arch=compute_50,code=sm_50")
ENDIF()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${NVCC_FLAGS_EXTRA};-O3 -lcublas -lcudart")

MESSAGE("NVCC FLAGS: " ${CUDA_NVCC_FLAGS})

IF(APPLE)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -stdlib=libstdc++ -Xlinker -stdlib=libstdc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
ENDIF()

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/TH" "${Torch_INSTALL_INCLUDE}/THC"
"${CMAKE_CURRENT_SOURCE_DIR}/cuda_helpers/")

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src
  cudaconv3/src/conv_util.cu
  cudaconv3/src/filter_acts.cu
  cudaconv3/src/img_acts.cu
  cudaconv3/src/weight_acts.cu
  bias_kernels.cu)

FILE(GLOB luasrc *.lua)

CUDA_ADD_LIBRARY(ccn2 MODULE ${src})
TARGET_LINK_LIBRARIES(ccn2 THC TH)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(ccn2 PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS ccn2
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/ccn2")
