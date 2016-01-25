require('..')
function testBiSequencer()
   local hiddenSize = 4
   local batchSize = 4
   local nStep = 3
   local fwd = nn.LSTM(hiddenSize, hiddenSize)
   local bwd = nn.LSTM(hiddenSize, hiddenSize)
   fwd:zeroGradParameters()
   bwd:zeroGradParameters()
  -- local brnn = nn.BiSequencer(fwd:clone('weight','bias','gradWeight','gradBias'), bwd:clone('weight','bias','gradWeight','gradBias'))
  -- local brnn2 = nn.BiSequencer(fwd:clone('weight','bias','gradWeight','gradBias'), bwd:clone('weight','bias','gradWeight','gradBias'))
   local brnn = nn.BiSequencer(fwd:clone('weight','bias'), bwd:clone('weight','bias')):cuda()
   local brnn2 = nn.BiSequencer(fwd:clone('weight','bias'), bwd:clone('weight','bias')):cuda()
	brnn.forwardModule:share(brnn2.forwardModule,'weight','bias')
	brnn.backwardModule:share(brnn2.backwardModule,'weight','bias')
   local inputs, gradOutputs = {}, {}
   local inputs2, gradOutputs2 = {}, {}
	for loop =1,3 do
	print(loop)
	nStep = nStep + loop
   for i=1,nStep do
      inputs[i] = torch.randn(hiddenSize):cuda()
      inputs2[i] = torch.randn(hiddenSize):cuda()
      gradOutputs[i] = torch.randn(hiddenSize*2):cuda()
      gradOutputs2[i] = torch.randn(hiddenSize*2):cuda()
   end
   local outputs = brnn:forward(inputs)
   local outputs2 = brnn2:forward(inputs2)
   local gradInputs = brnn:backward(inputs, gradOutputs)
   local gradInputs2= brnn2:backward(inputs2, gradOutputs2)
   -- params
   local params, gradParams = brnn:parameters()
   local params2, gradParams2 = brnn2:parameters()
   
   -- updateParameters
   brnn:updateParameters(0.1)
   brnn2:updateParameters(0.1)
   brnn:zeroGradParameters()
   brnn2:zeroGradParameters() 
--[[
	for i,param in pairs(params) do
		if i< 2 then
			print(param)
			print(params2[i])
		end
   end

	for i,param in pairs(params) do
		local tmp  = (param  + params2[i])/2
		params[i] = tmp
		params2[i] = tmp
	end
--]]
   for i,param in pairs(params) do
		if i< 2 then
			print(param)
			print(params2[i])
		end
   end
	print('------------')
	end
end

testBiSequencer()
