local C = ccn2.C

local SpatialConvolution, parent = torch.class('ccn2.SpatialConvolution', 'nn.Module')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kH, dH, padding)
   parent.__init(self)

   dH = dH or 1 -- stride
   padding = padding or 0

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kH = kH
   self.dH = dH
   self.padding = padding

   self.weight = torch.Tensor(nInputPlane*kH*kH, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane*kH*kH, nOutputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()

   self:reset()
   self:cuda()
end

function SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kH*self.kH*self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)   
end

local function typecheck(i)
   if torch.type(i) ~= 'torch.CudaTensor' then 
      error('Input is expected to be torch.CudaTensor') 
   end
end

function SpatialConvolution:updateOutput(input)
   typecheck(input)
   local nBatch = input:size(4)
   local oH = math.floor((self.padding * 2 + input:size(2) - self.kH) / self.dH + 1);
   local inputCollapsed = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
   self.output:resize(self.nOutputPlane*oH*oH, nBatch);

   C['convFilterActs'](inputCollapsed:cdata(), 
                       self.weight:cdata(), 
                       self.output:cdata(), 
                       input:size(2), oH, oH, 
                          -self.padding, self.dH, 
                       self.nInputPlane, 1);
   return self.output:view(self.nOutputPlane, oH, oH, nBatch)
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   typecheck(input); typecheck(gradOutput);
   if self.gradInput then
      return input.nn.SpatialConvolution_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   typecheck(input); typecheck(gradOutput);
   return input.nn.SpatialConvolution_accGradParameters(self, input, gradOutput, scale)
end