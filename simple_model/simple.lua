require('..')
---------------------------
function get_model()
--[[
	--词嵌入以及权值加成的计算
	local vec_dict,vecs = deep_cqa.get_sub_embedding()
	deep_cqa.emd_dict =	vec_dict
	deep_cqa.emd_vecs = vecs
	local emd_layer = nn.LookupTable(vecs:size(1),deep_cqa.config.emd_dim)
	emd_layer.weight:copy(vecs)
	local avg_emd = deep_cqa.AvgEmd()
--]]
	--------------------------
	--			margin 误差统计			-->	误差计算criterion
	--			diff(1)作差				--> 共享层（误差是否能够传递通过这层）
	--	sigmoid()			sigmoid()
	--	线性(1)				线性(1)		--> 共享权重
	--	并行(600)			并行(600)	--> 共享权重
	local qst = nn.Identity()()		--问题的向量均值表达（也可以是前一层采用了RNN之类的方法实现） 300维
	local t_ans = nn.Identity()()	--正确答案的向量表达（同上） 300维
	local f_ans = nn.Identity()()	--错误。。。（同上）

	local t_rep = nn.JoinTable(1)({qst,t_ans})	--两个向量做拼接，600维
	local f_rep = nn.JoinTable(1)({qst,f_ans})	--同上

	local t_prob = nn.Linear(600,1)(t_rep)	--转换成为预测是否为正确答案的标量	
	local f_prob = nn.Linear(600,1)(f_rep)	--转换成为预测是否为错误答案的标量
	share_params(t_prob,f_prob)

	local t_sig = nn.Sigmoid()(t_prob)	--加一个sigmoid函数，约束值的范围
	local f_sig = nn.Sigmoid()(f_prob)
	
--	local t_prob_2 = nn.Linear(30,1)(t_sig)	--转换成为预测是否为正确答案的标量	
--	local f_prob_2 = nn.Linear(30,1)(f_sig)	--转换成为预测是否为错误答案的标量
--	share_params(t_prob_2,f_prob_2)

--	local t_sig_2 = nn.Sigmoid()(t_prob_2)	--加一个sigmoid函数，约束值的范围
--	local f_sig_2 = nn.Sigmoid()(f_prob_2)


	local sub  = nn.CSubTable()({t_sig,f_sig})	--作差，优化的目标是使这个差逼近一个margin
	local simple = nn.gModule({qst,t_ans,f_ans},{sub})
	
	return simple
end
---------------------------
Simple = {}
Simple.vecs = nil
Simple.dict = nil
Simple.emd_layer = nil
Simple.avg =nil
function get300(sent)
	if Simple.dict == nil then
		Simple.dict,Simple.vecs = deep_cqa.get_sub_embedding()
		Simple.emd_layer = nn.LookupTable(Simple.vecs:size(1),deep_cqa.config.emd_dim)
		Simple.emd_layer.weight:copy(Simple.vecs)
		Simple.vecs =nil
		Simple.avg = deep_cqa.AvgEmd()
	end
	local idx = deep_cqa.read_one_sentence(sent,Simple.dict)
	local vecs = Simple.emd_layer:forward(idx)
	local ans = Simple.avg:forward(vecs)	
	return ans
end
---------------------------
function demo()
	local sents = {'... the is a simple text file','how is the sun today','what happens last night'}
	local simple = get_model()
	local vecs = {}
	for i =1,#sents do
		vecs[i] = get300(sents[i])
	end
	local criterion = nn.MarginCriterion(0.3)
	local pred = simple:forward(vecs)
	y = torch.Tensor(1)
	y[1]= 1
	local err =criterion:forward(pred,y)
	print(pred,err)
end
----------------------------
function train()
	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local simple = get_model()
	local criterion = nn.MarginCriterion(0.3)
	local params = nil
	local grad_params =nil
	local y =torch.Tensor(1)
	y[1] = 1
	params, grad_params = simple:getParameters()
	local optim_state = { learningRate = 0.05}
---------------------------------------------------------	
	local c_count= 0
	for i = train_set.size-1000,train_set.size do
			local idx = indices[i] --乱序后的样本
			local sample = train_set[idx]
			local vecs ={}
			for j=1 ,#sample do
				vecs[j] = get300(sample[j]):clone()
			end
			local pred = simple:forward(vecs)
			if pred[1] > 0.0 then
				c_count = c_count +1
			end			
	end
	print(c_count/1000*1.0)
	for i =train_set.size-1000,train_set.size do
		local feval = function(x)
			grad_params:zero()
			local loss = 0
			local idx = indices[i] --乱序后的样本
			local sample = train_set[idx]
			local vecs ={}
			for j=1 ,#sample do
				vecs[j] = get300(sample[j]):clone()
			end
			local pred = simple:forward(vecs)
			loss = criterion:forward(pred,y)
			local gold = torch.Tensor(1)
			gold[1] = 0.1
			local obj_grad = criterion:backward(pred,gold)
			local emd_grad = simple:backward(vecs,obj_grad)
			loss = loss + 1e-4*params:norm() ^ 2
			--grad_params:add(1e-4,params)
			return loss,grad_params
		end
		optim.adagrad(feval,params,optim_state)
		xlua.progress(i-train_set.size+1000,1000)
	end
	 c_count= 0
	for i = train_set.size-1000,train_set.size do
			local idx = indices[i] --乱序后的样本
			local sample = train_set[idx]
			local vecs ={}
			for j=1 ,#sample do
				vecs[j] = get300(sample[j]):clone()
			end
			local pred = directional Recurrent Neural Networks simple:forward(vecs)
			if pred[1] > 0 then
				c_count = c_count +1
			end			
	end
	print(c_count/1000*1.0)

end
train()



