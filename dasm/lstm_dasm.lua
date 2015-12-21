require('..')

function get_model()
	local lstm_config ={
		in_dim = 300,
		mem_dim = 30,
		number_layers = 1,
		gate_output = true,
	}
	local qst = nn.Identity()()
	local t_ans =nn.Identity()()
	local f_ans =nn.Identity()()
	--三个输入
	local q_lstm = deep_cqa.LSTM(lstm_config)(qst)	
	local t_lstm = deep_cqa.LSTM(lstm_config)(t_ans)	--输入300 输出30
	local f_lstm = deep_cqa.LSTM(lstm_config)(f_ans)	--输入300 输出30
	
	share_params(t_lstm,f_lstm)	--共享参数


	local reshape = nn.Reshape(30,1)(q_lstm)
	tsl_lstm = nn.Reshape(1,30)(t_lstm)
	fsl_lstm = nn.Reshape(1,30)(f_lstm)
	
	local tm = nn.MM()({reshape,tsl_lstm})
	local fm = nn.MM()({reshape,fsl_lstm})

	local sub = nn.CSubTable()({tm,fm})
	local norm = nn.SoftSign()(sub)
	local line = nn.Reshape(900)(norm)
	local linear = nn.Linear(900,1)(line)
	local model = nn.gModule({qst,t_ans,f_ans},{linear})
	return model
end
Dasm ={}
Dasm.vecs = nil
Dasm.dict = nil
Dasm.emd_layer = nil
function get_embeddings(sent)
	if Dasm.dict == nil then
		Dasm.dict,Dasm.vecs = deep_cqa.get_sub_embedding()
		Dasm.emd_layer = nn.LookupTable(Dasm.vecs:size(1),deep_cqa.config.emd_dim)
		Dasm.emd_layer.weight:copy(Dasm.vecs)
		Dasm.vecs = nil
	end
	local idx =deep_cqa.read_one_sentence(sent,Dasm.dict)
	local vec =Dasm.emd_layer:forward(idx)
	return vec
end

function train()
	local model = get_model()
	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(1)
	local params = nil
	local grad_params =nil
	local y =torch.Tensor({1})
	params, grad_params = model:getParameters()
	local optim_state = {learningRate = 0.05 }
-----------------------
	for i = 1 , train_set.size/20 do
		local feval = function (x)
			grad_params:zero()
			local sample = train_set[indices[i]]	--乱序选取
			local vecs ={}
			for j= 1,#sample do
				vecs[j] = get_embeddings(sample[j]):clone()
				vecs[j] = vecs[j]:chunk(vecs[j]:size(1),1)
				--print(vecs[j]:size(1))
			end
			y[1] = 1
			if i%2 ==1 then 
				local tmp =vecs[2]
				vecs[2] = vecs[3]
				vecs[3] = tmp
				y[1] = -1
			end
			local pred = model:forward(vecs)
			local loss = criterion:forward(pred,y)
			local obj_grad = criterion:backward(pred,y)
			local emd_grad = model:backward(vecs,obj_grad)
			loss = loss + 1e-4*params:norm()^2
			return loss,grad_params
		end
		optim.adagrad(feval,params,optim_state)
		xlua.progress(i,train_set.size/20)
	end
	print('test')
--------------------------
	local c_count = 0
	for i = 1 , train_set.size/20 do
		local sample = train_set[indices[i]]	--乱序选取
		local vecs ={}
		for j= 1,#sample do
				vecs[j] = get_embeddings(sample[j]):clone()
				vecs[j] = vecs[j]:chunk(vecs[j]:size(1),1)
		end
		local pred = model:forward(vecs)
		if pred[1][1] > 0 then
			c_count = c_count+1
		end
		xlua.progress(i,train_set.size/20)
			--local loss = criterion:forward(pred,y)
	end
	print(c_count*1.0/(train_set.size/20))
end

train()
