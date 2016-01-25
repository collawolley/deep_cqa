require('..')
function testBiSequencer()
   local hiddenSize = 8
   local batchSize = 4
   local nStep = 3
   local fwd = nn.LSTM(hiddenSize, hiddenSize)
   local bwd = nn.LSTM(hiddenSize, hiddenSize)
	fwd:share(bwd,'weight','bias','gradWeight','gradBias')
   fwd:zeroGradParameters()
   bwd:zeroGradParameters()
   local brnn = nn.BiSequencer(fwd:clone('weight','bias','gradWeight','gradBias'), bwd:clone('weight','bias','gradWeight','gradBias'))
   local inputs, gradOutputs = {}, {}
   for i=1,nStep do
      inputs[i] = torch.randn(batchSize, hiddenSize)
      gradOutputs[i] = torch.randn(batchSize, hiddenSize*2)
   end
   local outputs = brnn:forward(inputs)
   local gradInputs = brnn:backward(inputs, gradOutputs)
--   mytester:assert(#inputs == #outputs, "BiSequencer #outputs error")
  -- mytester:assert(#inputs == #gradInputs, "BiSequencer #outputs error")
   
   -- forward
   local fwdSeq = nn.Sequencer(fwd)
   local bwdSeq = nn.Sequencer(bwd)
   local zip, join = nn.ZipTable(), nn.Sequencer(nn.JoinTable(1,1))
   local fwdOutputs = fwdSeq:forward(inputs)
   local bwdOutputs = _.reverse(bwdSeq:forward(_.reverse(inputs)))
   local zipOutputs = zip:forward{fwdOutputs, bwdOutputs}
   local outputs2 = join:forward(zipOutputs)
   for i,output in ipairs(outputs) do
  --    mytester:assertTensorEq(output, outputs2[i], 0.000001, "BiSequencer output err "..i)
   end
   
   -- backward
   local joinGradInputs = join:backward(zipOutputs, gradOutputs)
   local zipGradInputs = zip:backward({fwdOutputs, bwdOutputs}, joinGradInputs)
   local bwdGradInputs = _.reverse(bwdSeq:backward(_.reverse(inputs), _.reverse(zipGradInputs[2])))
   local fwdGradInputs = fwdSeq:backward(inputs, zipGradInputs[1])
   local gradInputs2 = zip:forward{fwdGradInputs, bwdGradInputs}
   for i,gradInput in ipairs(gradInputs) do
      local gradInput2 = gradInputs2[i]
      gradInput2[1]:add(gradInput2[2])
    --  mytester:assertTensorEq(gradInput, gradInput2[1], 0.000001, "BiSequencer gradInput err "..i)
   end
   
   -- params
   local brnn2 = nn.Sequential():add(fwd):add(bwd)
   local params, gradParams = brnn:parameters()
   local params2, gradParams2 = brnn2:parameters()
   local params3, gradParams3 = fwd:parameters()
   local params4, gradParams4 = bwd:parameters()
  -- mytester:assert(#params == #params2, "BiSequencer #params err")
  -- mytester:assert(#params == #gradParams, "BiSequencer #gradParams err")
   for i,param in pairs(params) do
     -- mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencer params err "..i)
     -- mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencer gradParams err "..i)
   end
   
   -- updateParameters
   brnn:updateParameters(0.1)
   brnn2:updateParameters(0.1)
   brnn:zeroGradParameters()
   brnn2:zeroGradParameters()
   for i,param in pairs(params3) do
		print(param)
		print(params4[i])
     -- mytester:assertTensorEq(param, params2[i], 0.000001, "BiSequencer params update err "..i)
     -- mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, "BiSequencer gradParams zero err "..i)
   end
end

testBiSequencer()
