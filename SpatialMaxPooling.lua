local C = ccn2.C

local SpatialMaxPooling, parent = torch.class('ccn2.SpatialMaxPooling', 'nn.Module')

function SpatialMaxPooling:__init(kW, dW)
  parent.__init(self)
  
  self.kW = kW
  self.dW = dW
  
  self.output = torch.Tensor()
  
  self:cuda()
end

function SpatialMaxPooling:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local outputX = math.floor((input:size(2) - self.kW)/self.dW + 1)
  ccn2.C['convLocalMaxPool'](inputC:cdata(), self.output:cdata(), input:size(1), self.kW, 0, self.dW, outputX)
  
  local ims = math.sqrt(self.output:size(1)/input:size(1))
  self.output = self.output:view(input:size(1), ims, ims, nBatch)
  return self.output
end
  
