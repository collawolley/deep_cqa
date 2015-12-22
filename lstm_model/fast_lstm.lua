--[[
	使用Tree-LSTM模型的LSTM代码实现正确答案和错误答案的分类	
	author:	liangjz
	time:	2015-12-21
--]]
require('..')
-------------------------
--定义一些全局变量
FastM = {}
FastM.vecs = nil
FastM.dict = nil
FastM.emd = nil
FastM.lstm_config ={
	in_dim = 300,
	mem_dim = 30,
	num_layers = 1,
	gate_output = true,	
}
-------------------------
function get_index(sent)
	--	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	if FastM.dict == nil then
		--	载入字典和词向量查询层
		FastM.dict, FastM.vecs = deep_cqa.get_sub_embedding()
		FastM.emd = nn.LookupTable(FastM.vecs:size(1),deep_cqa.config.emd_dim)
		FastM.emd.weight:copy(FastM.vecs)
		FastM.vecs = nil
	end
	local index = deep_cqa.read_one_sentence(sent,FastM.dict)
		--	使用这个字典进行查询
	return index
end
--------------------------
function get_emd(index)
	--	备用函数，输入词的index组，输出查询到的词向量
	if FastM.emd == nil then 
		get_index('good start')
	end
	return  FastM.emd:forward(index)
end
-------------------------
function a_lstm()
	--	生成一个LSTM model
	return deep_cqa.LSTM(FastM.lstm_config)
end
-------------------------
function sim_model()
	--	先测试一下LSTM的速度
	local qst= nn.Identity()()
	local tans= nn.Identity()()
	local fans= nn.Identity()()

	local mem_dim = FastM.lstm_config.mem_dim
	
	local rsp_qst = nn.Reshape(mem_dim,1)(qst)
	local rsp_tans = nn.Reshape(1,mem_dim)(tans)
	local rsp_fans = nn.Reshape(1,mem_dim)(fans)
	
	local tm =nn.MM()({rsp_qst,rsp_tans})
	local fm =nn.MM()({rsp_qst,rsp_fans})

	local sub = nn.CSubTable()({tm,fm})
	local norm = nn.SoftSign()(sub)
	local reshape = nn.Reshape(mem_dim*mem_dim)(norm)
	local linear = nn.Linear(mem_dim*mem_dim,1)(reshape)
	local model = nn.gModule({qst,tans,fans},{linear})
	
	return model
end

function train()
	local qst_lstm = a_lstm()
	local tans_lstm = a_lstm()
	local fans_lstm = a_lstm()
	share_params(tans_lstm,fans_lstm)
	local model =sim_model()
	local modules = nn.Parallel():add(qst_lstm):add(tans_lstm):add(fans_lstm):add(model)
-------------------
	params,grad_params = modules:getParameters()

	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1})
	local batch_size = 10
	local optim_state = {learningRate = 0.05 }
	for i= 1,train_set.size,batch_size do
		local size = math.min(i+batch_size-1,batch_size)-i+1
		
		local feval = function(x)
			grad_params:zero()
			local loss = 0	--累积的损失
			
			for j = 1,size do
				xlua.progress(i+j-1,train_set.size)
				local idx = indices[i+j-1]
				local sample = train_set[idx]
				local vecs={}
				for k =1,#sample do
					local index = get_index(sample[k])
					vecs[k] =get_emd(index)
				end

				local qrep = qst_lstm:forward(vecs[1])
				local trep = tans_lstm:forward(vecs[2])
				local frep = fans_lstm:forward(vecs[3])
				if(idx %2 ==0) then
					trep,frep = frep,trep
					gold[1] = -1
				else
					gold[1] = 1
				end
				local pred = model:forward({qrep,trep,frep})
				local loss = loss + criterion:forward(pred,gold)
				
				local obj_grad = criterion:backward(pred,gold)
				local lstm_grad = model:backward({qrep,trep,frep},obj_grad)

				local qlstm_grad  = lstm_back(sample[1],vecs[1],lstm_grad[1],qst_lstm)
				local tlstm_grad  = lstm_back(sample[2],vecs[2],lstm_grad[2],qst_lstm)
				local flstm_grad  = lstm_back(sample[3],vecs[3],lstm_grad[3],qst_lstm)
			end
			grad_params = grad_params/size
			loss = loss / size
			loss = loss + 1e-4*params:norm()^2
			return loss,grad_params		
		end
		optim.adagrad(feval,params,optim_state)
	end
end

function lstm_back(sent,inputs,lstmgrad,lstm)
	local grad
	local config= FastM.lstm_config
	if config.num_layers == 1 then
		grad = torch.zeros(#inputs, config.mem_dim)
		grad[#inputs] = lstmgrad
	else
		grad = torch.zeros(#input, config.num_layers, config.mem_dim)
		for l = 1, config.num_layers do
			grad[{#inputs, l, {}}] = lstmgrad[l]
		end
	end
	local input_grads = lstm:backward(inputs, grad)
	return input_grads
end

train()
