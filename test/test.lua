require 'ccn2'
require 'cunn'

local ccntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}

function ccntest.SpatialConvolution_forward_batch()
    local bs = math.random(1,4) * 32
    local from = math.random(1,3); 
    if math.random(1,2) == 2 then 
       from = 16 * math.random(1,8)
    end
    local to = math.random(1,8) * 32
    local ki = math.random(3,15)
    local kj = ki
    local si = 1 -- not supported by CPU version yet
    local sj = si
    local outi = math.random(1,64)
    local outj = outi
    local ini = (outi-1)*si+ki
    local inj = ini

    local tm = {}
    local title = string.format('ccn2.SpatialConvolution.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]', 
                                bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
    times[title] = tm

    local input = torch.randn(from,inj,ini,bs):cuda()
    local sconv = nn.SpatialConvolutionCUDA(from,to,ki,kj,si,sj):cuda()
    local groundtruth = sconv:forward(input)
    local a = torch.Timer()
    for i = 1,nloop do
       groundtruth = sconv:forward(input)
    end
    tm.cpu = a:time().real

    local gconv = ccn2.SpatialConvolution(from,to,ki,si):cuda()
    gconv.weight:copy(sconv.weight)
    gconv.bias:copy(sconv.bias)
    local rescuda = gconv:forward(input)
    a:reset()
    for i = 1,nloop do
       rescuda = gconv:forward(input)
    end
    cutorch.synchronize()
    tm.gpu = a:time().real

    local error = rescuda:add(groundtruth:mul(-1))
    mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
 end

 function ccntest.SpatialConvolution_backward_batch()
    local bs = math.random(1,4) * 32
    local from = math.random(1,3); 
    if math.random(1,2) == 2 then 
       from = 16 * math.random(1,8)
    end
    local to = math.random(1,8) * 32
    local ki = math.random(3,15)
    local kj = ki
    local si = 1 -- not supported by CPU version yet
    local sj = si
    local outi = math.random(1,64)
    local outj = outi
    local ini = (outi-1)*si+ki
    local inj = ini
    local tm = {}
    local backwardScale = math.random(1, 10)/10
    local doPartialSum = math.random(0,1)
    local partialSum
    if doPartialSum == 1 then
       partialSum = math.random(1,6)
    end
    local title = string.format('ccn2.SpatialConvolution.backward(scale: %.1f, partialSum: %d) %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d', 
                                backwardScale, partialSum or -1, bs, from, inj, ini, kj, ki, bs, to, outj, outi)
    times[title] = tm

    local input = torch.randn(from,inj,ini,bs):cuda()
    local gradOutput = torch.randn(to,outj,outi,bs):cuda()
    local sconv = nn.SpatialConvolutionCUDA(from,to,ki,kj,si,sj):cuda()
    sconv:forward(input)
    sconv:zeroGradParameters()
    local groundgrad = sconv:backward(input, gradOutput)
    local a = torch.Timer()
    for i = 1,nloop do
       sconv:zeroGradParameters()
       groundgrad = sconv:backward(input, gradOutput, backwardScale)
    end
    local groundweight = sconv.gradWeight
    local groundbias = sconv.gradBias
    tm.cpu = a:time().real

    local gconv = ccn2.SpatialConvolution(from,to,ki,si, 0, 1, partialSum):cuda()
    gconv.weight:copy(sconv.weight)
    gconv.bias:copy(sconv.bias)
    gconv:forward(input)
    gconv:zeroGradParameters()
    local rescuda = gconv:backward(input, gradOutput)
    a:reset()
    for i = 1,nloop do
       gconv:zeroGradParameters()
       rescuda = gconv:backward(input, gradOutput, backwardScale)
    end
    local weightcuda = gconv.gradWeight
    local biascuda = gconv.gradBias
    cutorch.synchronize()
    tm.gpu = a:time().real

    local error = rescuda:add(groundgrad:mul(-1))
    local werror = weightcuda:add(groundweight:mul(-1))
    local berror = biascuda:add(groundbias:mul(-1))

    mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
    mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
    mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
 end

function ccntest.SpatialMaxPooling_forward_batch()
  local bs = math.random(1,4) * 32
  local from = math.random(1,3)
  local inj = math.random(1,64)
  local ini = inj
  
  local kw = 3
  local dw = 2
  
  tm = {}
  local title = string.format('ccn2.SpatialMaxPooling.forward %dx%dx%dx%d o %dx%d', 
                                bs, from, inj, ini, kw, dw)
  times[title] = tm
                              
  local input = torch.randn(from,inj,ini,bs):cuda()
  
  local spool = nn.SpatialMaxPoolingCUDA(kw, kw, dw, dw):cuda()
  local groundtruth = spool:forward(input)
  local a = torch.Timer()
  for i = 1,nloop do
    groundtruth = spool:forward(input)
  end
  cutorch.synchronize()
  tm.cpu = a:time().real
  
  local gpool = ccn2.SpatialMaxPooling(kw, dw):cuda()
  local rescuda = gpool:forward(input)
  a:reset()
  for i = 1,nloop do
    rescuda = gpool:forward(input)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real

  mytester:asserteq(groundtruth:size(2), rescuda:size(2), 'output size')
  mytester:asserteq((groundtruth:float() - rescuda:float()):max(), 0, 'error forward')
end


function ccntest.SpatialMaxPooling_backward_batch()
  local bs = 32
  local from = 16 * math.random(1,3)
  local to = from
  local ki = math.random(2,4)
  local kj = ki
  local si = ki
  local sj = kj
  local outi = math.random(16,32)
  local outj = outi
  local ini = (outi-1)*si+ki
  local inj = (outj-1)*sj+kj

  local tm = {}
  local title = string.format('ccn2.SpatialMaxPooling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d', 
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
  times[title] = tm

  local input = torch.randn(bs,from,inj,ini)
  local gradOutput = torch.randn(bs,to,outj,outi)
  input = input:resize(bs,from*ini*inj):t():contiguous():resize(from,ini,inj,bs):cuda()
  gradOutput = gradOutput:resize(bs,to*outi*outj):t():contiguous():resize(to,outi,outj,bs):cuda()
  local sconv = nn.SpatialMaxPoolingCUDA(ki,kj,si,sj):cuda()
  sconv:forward(input)
  sconv:zeroGradParameters()
  local groundgrad = sconv:backward(input, gradOutput)
  local a = torch.Timer()
  for i = 1,nloop do
    sconv:zeroGradParameters()
    groundgrad = sconv:backward(input, gradOutput)
  end
  tm.cpu = a:time().real
   
  local gconv = ccn2.SpatialMaxPooling(ki, si):cuda()
  gconv:forward(input)
  gconv:zeroGradParameters()
  rescuda = gconv:backward(input, gradOutput)
  a:reset()
  for i = 1,nloop do
    gconv:zeroGradParameters()
    rescuda = gconv:backward(input, gradOutput)
  end
  tm.gpu = a:time().real
   
  mytester:asserteq((groundgrad:float()-rescuda:float()):max(), 0, 'error backward')
end


function ccntest.SpatialAvgPooling_forward_batch()
  local bs = math.random(1,4) * 32
  local from = math.random(1,3)
  local inj = math.random(1,32)*2
  local ini = inj
  
  local kw = 2
  local dw = 2
  
  tm = {}
  local title = string.format('ccn2.SpatialAvgPooling.forward %dx%dx%dx%d o %dx%d', 
                                bs, from, inj, ini, kw, dw)
  times[title] = tm
  tm.cpu = 1
                              
  local input = torch.randn(from,inj,ini,bs):cuda()
  local a = torch.Timer()
  local gpool = ccn2.SpatialAvgPooling(kw, dw):cuda()
  local rescuda = gpool:forward(input)
  a:reset()
  for i = 1,nloop do
    rescuda = gpool:forward(input)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real

  mytester:assertlt(math.abs(input:sum()/(kw*kw) - rescuda:sum()), 1e-4, 'sum error')
end


function ccntest.SpatialAvgPooling_backward_batch()
  local bs = math.random(1,4) * 32
  local from = math.random(1,3)*16
  local inj = math.random(1,32)*2
  local ini = inj
  
  local kw = 2
  local dw = 2
  
  tm = {}
  local title = string.format('ccn2.SpatialAvgPooling.backward %dx%dx%dx%d o %dx%d', 
                                bs, from, inj, ini, kw, dw)
  times[title] = tm
  tm.cpu = 1
                              
  local input = torch.randn(from,inj,ini,bs):cuda()
  local a = torch.Timer()
  local gpool = ccn2.SpatialAvgPooling(kw, dw):cuda()
  local output = gpool:forward(input)
  local rescuda = gpool:backward(input, output)
  a:reset()
  for i = 1,nloop do
    rescuda = gpool:backward(input, output)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real

  -- TODO: add a real check
end

function ccntest.SpatialCrossResponseNormalization_batch()
    local bs = math.random(1,2) * 32
    local fmaps = 16 * math.random(1,4)
    local ini = math.random(5,17)
    local inj = ini
    local size = math.random(1,fmaps)
    local addScale = math.random()
    local powScale = math.random()
    local minDiv = math.random(1,2)
    
    local tm = {}
    local title = string.format('ccn2.SpatialCrossResponseNormalization.forward %dx%dx%dx%d [s: %d]'
                                , bs, fmaps, inj, ini, size, addScale, powScale, minDiv)
    times[title] = tm; tm.cpu = 1; tm.gpu = 1;

    local input = torch.randn(fmaps,inj,ini,bs):cuda()
    local mod = ccn2.SpatialCrossResponseNormalization(size, addScale, powScale, minDiv):cuda()
    local errmax, errmean = jac.testJacobian(mod, input)
    cutorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end

function ccntest.SpatialResponseNormalization_batch()
    local bs = math.random(1,2) * 32
    local fmaps = 16 * math.random(1,4)
    local ini = math.random(5,17)
    local inj = ini
    local size = math.random(1,ini)
    local addScale = math.random()
    local powScale = math.random()
    local minDiv = math.random(1,2)
    
    local tm = {}
    local title = string.format('ccn2.SpatialResponseNormalization.forward %dx%dx%dx%d [s: %d]'
                                , bs, fmaps, inj, ini, size, addScale, powScale, minDiv)
    times[title] = tm; tm.cpu = 1; tm.gpu = 1;

    local input = torch.randn(fmaps,inj,ini,bs):cuda()
    local mod = ccn2.SpatialResponseNormalization(size, addScale, powScale, minDiv):cuda()
    local errmax, errmean = jac.testJacobian(mod, input)
    cutorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end

function ccntest.SpatialConvolutionLocal_batch()
    local bs = math.random(1,2) * 32
    local from = math.random(1,3)
    local to = math.random(1,2) * 32
    local ki = math.random(3,15)
    local si = 1 -- not supported by CPU version yet
    local outi = math.random(1,20)
    local ini = (outi-1)*si+ki

    local tm = {}
    local title = string.format('ccn2.SpatialConvolutionLocal.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
        bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    times[title] = tm
    tm.cpu = 1
    tm.gpu = 1

    local input = torch.randn(from,ini,ini,bs):cuda()
    local mod = ccn2.SpatialConvolutionLocal(from,to,ini,ki,si):cuda()
    local errmax, errmean = jac.testJacobian(mod, input)
    cutorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end

function ccntest.SpatialCrossMaxPooling_batch()
    local bs = math.random(1,2) * 32
    local fmaps = 16 * math.random(1,4)
    local ini = math.random(5,17)
    local inj = ini
    local kD = math.random(1,fmaps)
    local dD = math.random(1,fmaps)

    local tm = {}
    local title = string.format('ccn2.SpatialCrossMaxPooling %dx%dx%dx%d [kD: %d dD: %d]'
                                , bs, fmaps, inj, ini, kD, dD)
    times[title] = tm; tm.cpu = 1; tm.gpu = 1;

    local input = torch.randn(fmaps,inj,ini,bs):cuda()
    local mod = ccn2.SpatialCrossMaxPooling(kD, dD):cuda()
    local errmax, errmean = jac.testJacobian(mod, input)
    cutorch.synchronize()
    mytester:assertlt(errmax, precision_jac, 'Jacobian test failed!')
end


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
jac = ccn2.Jacobian
mytester = torch.Tester()
mytester:add(ccntest)
mytester:run(tests)
print ''
print ' ------------------------------------------------------------------------------------------------'
print '|  Module                                                                          |  Speedup    |'
print ' ------------------------------------------------------------------------------------------------'
for module,tm in pairs(times) do
   local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
   print(str)
end
print ' ------------------------------------------------------------------------------------------------'
