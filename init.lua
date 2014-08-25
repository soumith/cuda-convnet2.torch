require 'cutorch'
require 'nn'
ccn2 = {}
include 'ffi.lua'
include 'utils.lua'
include 'Jacobian.lua'

include 'SpatialConvolution.lua'
include 'SpatialMaxPooling.lua'
include 'SpatialContrastNormalization.lua'
include 'SpatialCrossResponseNormalization.lua'
