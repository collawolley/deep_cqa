require('..')

function get_model()
	local qst = nn.Identity()()
	local t_ans =nn.Identity()()
	local f_ans =nn.Identity()()
	--三个输入
	local q_lstm = nn.LSTM(300,50,9999)	--输入300 输出50 最长序列9999
	local t_lstm = nn.LSTM(300,50,9999)
	local f_lstm = nn.LSTM(300,50,9999)
	local qs_lstm = nn.Sequencer(q_lstm)(qst)
	local ts_lstm = nn.Sequencer(t_lstm)(t_ans)
	local fs_lstm = nn.Sequencer(f_lstm)(f_ans)
	
	share_params(ts_lstm,fs_lstm)	--共享参数
	local reshape = nn.Reshape(50,1)(qs_lstm)

	local tm = nn.MM()({reshape,ts_lstm})
	local fm = nn.MM()({reshape,fs_lstm})

	local sub = nn.CSubTable()({tm,fm})
	local norm = nn.SoftSign()(sub)
	local linear = nn.Linear(2500,1)(norm)
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
	local criterion = nn.MarginCriterion(0.5)
	local params = nil
	local grad_params =nil
	local y =torch.Tensor(1)
	y[1] = 1
	params, grad_params = model:getParameters()
	local optim_state = {learningRate = 0.05 }
-----------------------
	for i = 1 , train_set.size- 1000 do
		local feval = function (x)
			grad_params:zero()
			local sample = train_set[indices[i]]	--乱序选取
			local vecs ={}
			for j= 1,#sample do
				vecs[j] =get_embeddings(sample[j]):clone()
				print(j)
				print(sample[j])
				print(vecs[j]:size())
			end
			y[1] = 1
		--[[
			if i%2 ==1 then 
				local tmp =vecs[2]
				vecs[2] = vecs[3]
				vecs[3] = tmp
				y[1] = -1
			end
			--]]
			print(vecs[1]:size())
			print(vecs[2]:size())
			print(vecs[3]:size())
			local pred = model:forward(vecs)
			local loss = criterion:forward(pred,y)
			local obj_grad = criterion:backward(pred,y)
			local emd_grad = model:backward(vecs,obj_grad)
			loss = loss + 1e-4*params:norm()^2
			return loss,grad_params
		end
		optim.adagrad(feval,params,optim_state)
		xlua.progress(i,train_set.size-1000)
	end
--------------------------
	local c_count = 0
	for i = train_set.size-999 , train_set.size do
		local sample = train_set[indices[i]]	--乱序选取
		local vecs ={}
		for j= 1,#sample do
			vecs[j] =get_embeddings(sample[j])
		end
		local pred = model:forward(vecs)
		if pred > 0 then
			c_count = c_count+1
		end
			--local loss = criterion:forward(pred,y)
	end
	print(c_count*1.0/1000)
end

train()
