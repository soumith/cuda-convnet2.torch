local C = ccn2.C

local SpatialMaxPooling, parent = torch.class('ccn2.SpatialMaxPooling', 'nn.Module')

function SpatialMaxPooling:__init(kW, dW)
  parent.__init(self)

  self.kW = kW
  self.dW = dW or kW

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()

end

function SpatialMaxPooling:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local outputX = math.ceil((input:size(2) - self.kW)/self.dW + 1)

  C['convLocalMaxPool'](inputC:cdata(), self.output:cdata(), input:size(1), self.kW, 0, self.dW, outputX)

  local ims = math.sqrt(self.output:size(1)/input:size(1))
  self.output = self.output:view(input:size(1), ims, ims, nBatch)
  return self.output
end


function SpatialMaxPooling:updateGradInput(input, gradOutput)
  ccn2.typecheck(input); ccn2.typecheck(gradOutput);
  ccn2.inputcheck(input); ccn2.inputcheck(gradOutput);
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local gradOutputC = gradOutput:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))
  local outputC = self.output:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))

  self.gradInput:resize(inputC:size())
  local outputX = math.ceil((input:size(2) - self.kW)/self.dW + 1)

  C['convLocalMaxUndo'](inputC:cdata(), gradOutputC:cdata(), outputC:cdata(), self.gradInput:cdata(), self.kW, 0, self.dW, outputX)
  self.gradInput = self.gradInput:view(input:size(1), input:size(2), input:size(3), input:size(4))
  return self.gradInput
end
