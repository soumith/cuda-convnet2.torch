require 'ccn2'

local ccntest_jac = {}
local errs = {}

function ccntest_jac.SpatialConvolution_Jacobian()
    local bs = 32
    local from = 1
    local to = 32
    local ki = 3
    local si = 1 -- not supported by CPU version yet
    local outi = 4
    local ini = (outi-1)*si+ki

    local tm = {}
    local title = string.format('ccn2.SpatialConvolution %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                                bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    errs[title] = tm

    local module = ccn2.SpatialConvolution(from,to,ki,si):cuda()
    local input = torch.randn(from,ini,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    --mytester:assertlt(err,precision, 'error on state ')
    tm.err = err
end

function ccntest_jac.SpatialConvolutionLocal_Jacobian()
    local bs = 32
    local from = math.random(1,3)
    local to = 32
    local ki = 3
    local si = 1 -- not supported by CPU version yet
    local outi = 4
    local ini = (outi-1)*si+ki

    local tm = {}
    local title = string.format('ccn2.SpatialConvolutionLocal %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                                bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    errs[title] = tm

    local module = ccn2.SpatialConvolutionLocal(from,to,ini,ki,si):cuda()
    local input = torch.randn(from,ini,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    tm.err = err
end

function ccntest_jac.SpatialCrossResponseNormalization_Jacobian()
    local bs = 32
    local from = 16
    local inj = 2
    local ini = inj

    tm = {}
    local title = string.format('ccn2.SpatialCrossResponseNormalization %dx%dx%dx%d',
                                bs, from, inj, ini)
    errs[title] = tm

    local module = ccn2.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2)
    local input = torch.randn(from,inj,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    tm.err = err
end

math.randomseed(os.time())
jac = ccn2.Jacobian
mytester = torch.Tester()
mytester:add(ccntest_jac)
mytester:run(tests)
print ''
print ' ---------------------------------------------------------------------------------------------------'
print '|  Module                                                                          |  Jacobian err |'
print ' ---------------------------------------------------------------------------------------------------'
for module,tm in pairs(errs) do
    local str = string.format('| %-80s | %4.5f       |', module, tm.err)
    print(str)
end
print ' ---------------------------------------------------------------------------------------------------'
