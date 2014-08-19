local ffi = require 'ffi'

ffi.cdef[[
void convFilterActs(THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, 
                    int paddingStart, int moduleStride,
                    int numImgColors, int numGroups);
void convFilterActsSt(THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput);

void localFilterActs(THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups);
void localFilterActsSt(THCudaTensor* images, THCudaTensor* filters, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups,
                     float scaleTargets, float scaleOutput);

void convImgActs(THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void convImgActsSt(THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput);

void localImgActs(THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void localImgActsSt(THCudaTensor* hidActs, THCudaTensor* filters, THCudaTensor* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput);

void convWeightActs(THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                    int moduleStride, int numImgColors, int numGroups, int sumWidth);
void convWeightActsSt(THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, int sumWidth,
                    float scaleTargets, float scaleOutput);

void localWeightActs(THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                     int moduleStride, int numImgColors, int numGroups);

void localWeightActsSt(THCudaTensor* images, THCudaTensor* hidActs, THCudaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void addBias(THCudaTensor* output, THCudaTensor* bias);
void gradBias(THCudaTensor* output, THCudaTensor* gradBias, float scale);

void convLocalMaxPool(THCudaTensor* images, THCudaTensor* target, int numFilters,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalMaxUndo(THCudaTensor* images, THCudaTensor* maxGrads, THCudaTensor* maxActs, THCudaTensor* target,
                      int subsX, int startX, int strideX, int outputsX);

void convLocalAvgPool(THCudaTensor* images, THCudaTensor* target, int numFilters,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(THCudaTensor* avgGrads, THCudaTensor* target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

void convResponseNorm(THCudaTensor* images, THCudaTensor* denoms, THCudaTensor* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void convResponseNormUndo(THCudaTensor* outGrads, THCudaTensor* denoms, THCudaTensor* inputs, THCudaTensor* acts, THCudaTensor* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convContrastNorm(THCudaTensor* images, THCudaTensor* meanDiffs, THCudaTensor* denoms, THCudaTensor* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void convContrastNormUndo(THCudaTensor* outGrads, THCudaTensor* denoms, THCudaTensor* meanDiffs, THCudaTensor* acts, THCudaTensor* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);
]]

ccn2.C = ffi.load(package.searchpath('libccn2', package.cpath))