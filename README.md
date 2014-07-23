cuda-convnet2.torch
===================

Torch7 bindings for cuda-convnet2 kernels!


NVMatrix to THTensor cheatsheet
===============================

| NVMatrix            | THCudaTensor |
| --------------------|:-------------:|
| .getNumCols()       | .size[0]
| .getNumRows()       | .size[1]
| .isTrans()          | N/A
| .getDevData()       | THCudaTensor_data()
| .resize()           | THCudaTensor_resizeXd where X = dims
| .getTextureObject() | N/A
| .isContiguous       | THCudaTensor_isContiguous

* check contiguity of all tensors, if not, make contiguous
* ignore/remove assertions (because you are doing contiguous checks anyways)

Agg = ?, Agg.getBaseValue, Agg.output(.., ..)
