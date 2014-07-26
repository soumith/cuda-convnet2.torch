cuda-convnet2.torch
===================

Torch7 bindings for cuda-convnet2 kernels!
Kept as a separate repo because of the License, and because the codebase is not small.


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
