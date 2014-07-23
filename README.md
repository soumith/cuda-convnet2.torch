cuda-convnet2.torch
===================

Torch7 bindings for cuda-convnet2 kernels!

NVMatrix -> THCudaTensor*
.getNumCols() -> ->size[0]
.getNumRows() -> ->size[1]
check contiguity of all tensors, if not, make contiguous
!.isTrans() -> ignore/remove assertions (because you are doing contiguous checks anyways)
.getDevData() -> THCudaTensor_data()
.resize() -> THCudaTensor_resizeXd where X = number of dims
.getTextureObject() -> not supported yet by THTensor, but it is just a small function that transfers the tensor to texture memory.


Agg = ?, Agg.getBaseValue, Agg.output(.., ..)