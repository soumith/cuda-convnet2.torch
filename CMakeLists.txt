CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 5.5 REQUIRED)

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35 -O3 -lcublas -lcudart")

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