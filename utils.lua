function ccn2.typecheck(i)
   if torch.type(i) ~= 'torch.CudaTensor' then 
      error('Input is expected to be torch.CudaTensor') 
   end
end

function ccn2.inputcheck(i)
   -- square image
   if i:size(2) ~= i:size(3) then
      error('Assertion failed: [i:size(2) == i:size(3)]. Only square images supported by this module!')
   end
   if math.fmod(i:size(4), 32) ~= 0 then
      error('Assertion failed: [i:size(4) % 32 == 0]. Only batchSizes which are a multiple of 32 are supported.')
   end
end

