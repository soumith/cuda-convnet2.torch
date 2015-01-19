local ffi = require 'ffi'

ffi.cdef[[
void convFilterActs(THCState* state, THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX,
                    int paddingStart, int moduleStride,
                    int numImgColors, int numGroups);
void convFilterActsSt(THCState* state, THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput);

void localFilterActs(THCState* state, THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups);
void localFilterActsSt(THCState* state, THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups,
                     float scaleTargets, float scaleOutput);

void convImgActs(THCState* state, THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void convImgActsSt(THCState* state, THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput);

void localImgActs(THCState* state, THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void localImgActsSt(THCState* state, THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput);

void convWeightActs(THCState* state, THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                    int moduleStride, int numImgColors, int numGroups, int sumWidth);
void convWeightActsSt(THCState* state, THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, int sumWidth,
                    float scaleTargets, float scaleOutput);

void localWeightActs(THCState* state, THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                     int moduleStride, int numImgColors, int numGroups);

void localWeightActsSt(THCState* state, THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void addBias(THCState* state, THCudaTensor* output, THCudaTensor* bias);
void gradBias(THCState* state, THCudaTensor* output, THCudaTensor* gradBias, float scale);

void addSumCols(THCState* state, THCudaTensor*output, THCudaTensor*input); // used for partialSum

void convLocalMaxPool(THCState* state, THCudaTensor* images, THCudaTensor* target, int numFilters,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalMaxUndo(THCState* state, THCudaTensor* images, THCudaTensor* maxGrads, THCudaTensor* maxActs, THCudaTensor* target,
                      int subsX, int startX, int strideX, int outputsX);

void convCrossMapMaxPool(THCState* state, THCudaTensor* images, THCudaTensor* target, const int startF, const int poolSize,
                         const int numOutputs, const int stride, const int imgSize);
void convCrossMapMaxPoolUndo(THCState* state, THCudaTensor* images, THCudaTensor* maxGrads, THCudaTensor* maxActs, THCudaTensor* target,
                             const int imgSize, const int startF, const int poolSize,
                             const int stride, const float scaleTargets, const float scaleOutputs);

void convLocalAvgPool(THCState* state, THCudaTensor* images, THCudaTensor* target, int numFilters,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(THCState* state, THCudaTensor* avgGrads, THCudaTensor* target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

void convResponseNorm(THCState* state, THCudaTensor* images, THCudaTensor* denoms, THCudaTensor* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void convResponseNormUndo(THCState* state, THCudaTensor* outGrads, THCudaTensor* denoms, THCudaTensor* inputs, THCudaTensor* acts, THCudaTensor* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convContrastNorm(THCState* state, THCudaTensor* images, THCudaTensor* meanDiffs, THCudaTensor* denoms, THCudaTensor* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void convContrastNormUndo(THCState* state, THCudaTensor* outGrads, THCudaTensor* denoms, THCudaTensor* meanDiffs, THCudaTensor* acts, THCudaTensor* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convResponseNormCrossMap(THCState* state, THCudaTensor* images, THCudaTensor* target, int numFilters, int sizeF, float addScale,
                              float powScale, float minDiv, bool blocked);
void convResponseNormCrossMapUndo(THCState* state, THCudaTensor* outGrads, THCudaTensor* inputs, THCudaTensor* acts, THCudaTensor* target, int numFilters,
                         int sizeF, float addScale, float powScale, float minDiv, bool blocked, float scaleTargets, float scaleOutput);

void convResizeBilinear(THCState* state, THCudaTensor* images, THCudaTensor* target, int imgSize, int tgtSize, float scale);
]]

local path = package.searchpath('libccn2', package.cpath)
if not path then
   path = require 'ccn2.config'
end
assert(path, 'could not find libccn2.so')
ccn2.C = ffi.load(path)
