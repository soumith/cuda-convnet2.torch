cuda-convnet2.torch
===================

Torch7 bindings for cuda-convnet2 kernels!
Kept as a separate repo because of the License, and because the codebase is not small.

**This is a Work IN PROGRESS! **
**DONT USE any modules which are not listed below**

####Modules that are usable:
```
ccn2.SpatialConvolution(nInputPlane, nOutputPlane, kH, [dH = 1], [padding = 0], [groups = 1], [partialSum = oH * oH])
ccn2.SpatialConvolutionLocal(nInputPlane, nOutputPlane, inputHeight, kH, [dH = 1], [padding = 0])
ccn2.SpatialMaxPooling(kW, [dW = kW])
ccn2.SpatialAvgPooling(kW, [dW = kW])
ccn2.SpatialCrossResponseNormalization(nCrossFeaturemaps, [addScale = 0.0001], [powScale = 0.75], [minDiv = 1])
```

####What's left to do?

All the modules from here: https://code.google.com/p/cuda-convnet/wiki/LayerParams

####How to do it?
it is pretty simple, 
* Add the function signature from cudaconv3/include into ffi.lua
* Call the function in your lua module

For an example, look at SpatialConvolution.lua, 

#### How to use them?
Either send in an input of layout `Depth x Height x Width x Batch`, or wrap around `nn.Transpose` modules

Example
```lua
fSize = {3, 96, 128, 128, 384}
features = nn.Sequential()
features:add(nn.Transpose({1,4},{1,3},{1,2}))
features:add(ccn2.SpatialConvolution(fSize[1], fSize[2], 9))
features:add(nn.ReLU())
features:add(ccn2.SpatialMaxPooling(2,2))
features:add(ccn2.SpatialConvolution(fSize[2], fSize[3], 5))
features:add(nn.ReLU())
features:add(ccn2.SpatialMaxPooling(2,2))
features:add(ccn2.SpatialConvolution(fSize[3], fSize[4], 4))
features:add(nn.ReLU())
features:add(ccn2.SpatialConvolution(fSize[4], fSize[5], 3))
features:add(nn.ReLU())
features:add(ccn2.SpatialMaxPooling(2,2))
features:add(nn.Transpose({4,1},{4,2},{4,3}))
features:add(nn.Reshape(featuresOut))
```

###NVMatrix to THTensor cheatsheet
| NVMatrix            | THCudaTensor |
| --------------------|:-------------:|
| .getNumCols()       | .size[1]
| .getNumRows()       | .size[0]
| .getNumElements()   | THCudaTensor_nElement()
| .getNumDataBytes()  | THCudaTensor_nElement() * 4
| .getStride()        | .stride[0] 
| .isTrans()          | N/A
| .getDevData()       | THCudaTensor_data()
| .resize()           | THCudaTensor_resizeXd where X = dims
| .getTextureObject() | THCudaTensor_getTextureObject
| .isContiguous       | THCudaTensor_isContiguous
| .isSameDims         | THCudaTensor_isSameSizeAs
| .apply              | THCudaTensor_fill()

* check contiguity of all tensors, if not, make contiguous
* ignore/remove assertions (because you are doing contiguous checks anyways)
* harmonize getTextureObject. destroy all the texture objects after usage, treat them like pointers. NVMatrix does it in it's destructor, but since the object is not a member of the THCudaTensor structure, we have to destroy it manually after use.
* double-check places where strides are allowed (especially conv)
Agg = ?, Agg.getBaseValue, Agg.output(.., ..)
* Remember that NVMatrix only supports 2D tensors!
