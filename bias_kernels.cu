#include "nvmatrix/include/nvmatrix_kernels.cuh"
#include "nvmatrix/include/nvmatrix_operators.cuh"
#include <THC.h>
#include <algorithm>
#include "helper_cuda.h"

#define THAssert(exp)  if (exp) ; \
  else THError("assert(%s) failed in file %s, line %d", #exp, __FILE__, __LINE__);

extern "C" {
  
  void addBias(THCudaTensor* output, THCudaTensor* bias) {
    int width = output->size[1];
    int height = output->size[0];
    float *odata = THCudaTensor_data(output);
    float *bdata = THCudaTensor_data(bias);
    dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);
    dim3 blocks(std::min(512, DIVUP(width, ADD_VEC_THREADS_X)),
                std::min(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));
    kColVectorOp<NVMatrixBinaryOps::Add>
      <<<blocks, threads>>>(odata, bdata, odata, width, height, 
                            output->stride[0], output->stride[0], 
                            NVMatrixBinaryOps::Add());
    getLastCudaError("Kernel execution failed");
  }
  
  void gradBias(THCudaTensor* gradOutput, THCudaTensor* gradBias, float scale) {
    dim3 threads(AWR_NUM_THREADS);
    dim3 blocks(1, gradOutput->size[0]);
    kAggRows_wholerow_nosync<<<blocks, threads>>>(THCudaTensor_data(gradOutput), THCudaTensor_data(gradBias), gradOutput->size[1], gradOutput->size[0], NVMatrixAggs::Sum(), NVMatrixOps::Identity(), NVMatrixBinaryOps::SecondScaled(scale));
  }

  void addSumCols(THCudaTensor*output, THCudaTensor*input) { // output = weights, input = wtemp
    int width = input->size[1];
    int height = input->size[0];
    THCudaTensor_resize2d(output, 1, width);
    if ((height <= 2048 || width >= 4096)) {
      int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
      THAssert(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
      THAssert(numBlocks < NUM_BLOCKS_MAX);
      cudaTextureObject_t texInput = THCudaTensor_getTextureObject(input);
      kDumbAggCols<NVMatrixAggs::Sum, NVMatrixOps::Identity, NVMatrixBinaryOps::SecondScaled>
        <<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0>>>(texInput, 
                                                          THCudaTensor_data(output), width, height,
                                                          NVMatrixAggs::Sum(), NVMatrixOps::Identity(), 
                                                          NVMatrixBinaryOps::SecondScaled(1.0));
      getLastCudaError("kDumbAggCols: Kernel execution failed");
      checkCudaErrors(cudaDestroyTextureObject(texInput));
    } else { // Specialize the case when we have very long columns and few of them
      const int sumLength = 128;
      THCudaTensor* tmp = THCudaTensor_newWithSize2d(DIVUP(height, sumLength), width);
      int numBlocksX = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
      int numBlocksY = DIVUP(height, sumLength);
      dim3 blocks(numBlocksX, numBlocksY);
      dim3 threads(NUM_SUM_COLS_THREADS_PER_BLOCK);
      cudaTextureObject_t texInput = THCudaTensor_getTextureObject(input);
      kAggCols<NVMatrixAggs::Sum, NVMatrixOps::Identity>
        <<<blocks,threads, 0>>>(texInput, THCudaTensor_data(tmp), 
                                width, height, sumLength, NVMatrixAggs::Sum(), NVMatrixOps::Identity());
      getLastCudaError("kAggCols: Kernel execution failed");
      checkCudaErrors(cudaDestroyTextureObject(texInput));

      int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
      cudaTextureObject_t texTmp = THCudaTensor_getTextureObject(tmp);
      kDumbAggCols<NVMatrixAggs::Sum, NVMatrixOps::Identity, NVMatrixBinaryOps::SecondScaled>
        <<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK, 0>>>(texTmp, THCudaTensor_data(output), width, height,
                                                          NVMatrixAggs::Sum(), NVMatrixOps::Identity(), 
                                                          NVMatrixBinaryOps::SecondScaled(1.0));
      getLastCudaError("kDumbAggCols: Kernel execution failed");
      checkCudaErrors(cudaDestroyTextureObject(texTmp));
      THCudaTensor_free(tmp);
    }
  }
}
