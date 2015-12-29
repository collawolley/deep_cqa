--[[
	使用rnn的LSTM代码实现正确答案和错误答案的分类	
	author:	liangjz
	time:	2015-12-22
--]]
require('..')
local cfg = {}
cfg.vecs = nil
cfg.dict = nil
cfg.emd = nil
cfg.dim = deep_cqa.config.emd_dim
cfg.batch  = 10 or deep_cqa.config.batch_size
cfg.gpu = false
deep_cqa.ins_meth.load_binary()	--保险数据集，这里载入是为了获得测试集和答案
-----------------------
function get_index(sent)
	--	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	if cfg.dict == nil then
		--	载入字典和词向量查询层
		cfg.dict, cfg.vecs = deep_cqa.get_sub_embedding()
		cfg.emd = nn.LookupTable(cfg.vecs:size(1),cfg.dim)
		cfg.emd.weight:copy(cfg.vecs)
		cfg.vecs = nil
	end
	return deep_cqa.read_one_sentence(sent,cfg.dict)
end
------------------------------------------------------------------------
function getlm()
	get_index('today is')
-------------------------------------
	local qcov = nn.SpatialConvolution(1,1000,3,3,1,1,2,2)	--input需要是3维tensor
	local tcov = qcov:clone()
	local fcov = qcov:clone()
	share_params(qcov,tcov)	-- 权重共享，但是求导的过程各自独立
	share_params(qcov,fcov)	-- 更新参数在同一组变量上更新
-------------------------------------
	local pt = nn.Sequential()
	pt:add(nn.SpatialAdaptiveMaxPooling(1,1))
	pt:add(nn.Reshape(1000))
	pt:add(nn.Tanh())
-------------------------------------
	local hlq = nn.Linear(cfg.dim,200)	--词向量做变换
	local hlt = hlq:clone()
	local hlf = hlq:clone()
	share_params(hlq,hlt)
	share_params(hlq,hlf)

-------------------------------------
	local lm = {}	--待返回的语言模型
	lm.emd = cfg.emd	--词嵌入部分
	lm.qst = nn.Sequential()
	lm.tas = nn.Sequential()
	lm.fas = nn.Sequential()

	lm.qst:add(hlq)	--词嵌入的线性变换
	lm.tas:add(hlt)
	lm.fas:add(hlf)
	---------------------
	lm.qst:add(nn.Replicate(1))	--二维变三维
	lm.tas:add(nn.Replicate(1))
	lm.fas:add(nn.Replicate(1))
	---------------------
	lm.qst:add(qcov)	--卷积层
	lm.tas:add(tcov)
	lm.fas:add(fcov)
	---------------------
	lm.qst:add(pt:clone())	-- max 1 pooling & tanh
	lm.tas:add(pt:clone())
	lm.fas:add(pt:clone())
	----------------------
	lm.qt = nn.CosineDistance()	--cosine相似度
	lm.qf = nn.CosineDistance()	-- qt 应该越小越好，qf应该越大越好
	lm.sub = nn.PairwiseDistance(1)	-- 求距离之差，
-------------------------------
	if cfg.gpu then
		lm.emd:cuda()
		lm.qst:cuda()
		lm.tas:cuda()
		lm.fas:cuda()
		lm.qt:cuda()
		lm.qf:cuda()
		lm.sub:cuda()
	end
-------------------------------
	return lm

end
function testlm()	--应用修改模型后测试模型是否按照预期执行
	local lm = getlm()
	local criterion = nn.MarginCriterion(1)	--这个是magin的值
	local gold = torch.Tensor({1})	--gold是是否反转的意思,取值只能是+-1
	
	local index1 = get_index('today is a good day'):clone()
	local index2 = get_index('today is a very good day'):clone()
	local index3 = get_index('This class creates an output where the input is replicated'):clone()
	if cfg.gpu then
		criterion:cuda()
		gold =gold:cuda()
		index1 = index1:cuda()
		index2 = index2:cuda()
		index3 = index3:cuda()
	end
	local vec1 = lm.emd:forward(index1):clone()
	local vec2 = lm.emd:forward(index2):clone()
	local vec3 = lm.emd:forward(index3):clone()
	
--	local trans = nn.Transpose({1,2})
--print(trans:forward(vec1):size())
	local hl = nn.Linear(cfg.dim,200)
	local q =  lm.qst:forward(vec1)
	local t =  lm.tas:forward(vec2)
	local f =  lm.fas:forward(vec3)
	local qt = lm.qt:forward({q,t})
	local qf = lm.qf:forward({q,f})
	local sub = lm.sub:forward({qf,qt})
	print(qt,qf,sub)


end

cfg.lm = getlm()
-------------------------
function train()
	local lm = cfg.lm
	
	local modules = nn.Parallel()
	--modules:add(lm.emd)
	modules:add(lm.qst)
	modules:add(lm.tas)
	modules:add(lm.fas)
	modules:add(lm.tq)
	modules:add(lm.fq)
	modules:add(lm.sub)
	params,grad_params = modules:getParameters()	--用于参数更新

	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(0.5)
	local gold = torch.Tensor({1})
	if cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	local batch_size = cfg.batch
	local optim_state = {learningRate = 0.05,learningRateDecay = 0.02 }
	train_set.size =2000
	
	for i= 1,train_set.size,batch_size do
		local size = math.min(i+batch_size-1,train_set.size)-i+1
		local feval = function(x)
			grad_params:zero()
			lm.emd:zeroGradParameters()
			
			local loss = 0	--累积的损失
			for j = 1,size do
				xlua.progress(i+j-1,train_set.size)
				local idx = indices[i+j-1]
				local sample = train_set[idx]
				local sent_index = {}
				local vecs={}
				for k =1,#sample do
					sent_index[k] = get_index(sample[k]):clone()
					if(cfg.gpu) then sent_index[k] = index:cuda() end
					vecs[k] = lm.emd:forward(sent_index[k]):clone()
				end
				
				if(idx %2 ==0) then
					vecs[3],vecs[2] = vecs[2],vecs[3]
					gold[1] = -1	--翻转
				else
					gold[1] = 1		--正常不翻转
				end

				local rep1 = lm.qst:forward(vecs[1])
				local rep2 = lm.tas:forward(vecs[2])
				local rep3 = lm.fas:forward(vecs[3])
				

				local sc_1 = lm.qt:forward({rep1,rep2})
				local sc_2 = lm.qf:forward({rep1,rep3})
				local pred = lm.sub:forward({sc_2,sc_1})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
				
				local loss = loss + criterion:forward(pred,gold)
				
				local e1 = criterion:backward(pred,gold)
				local e2 = lm.sub:backward({sc_2,sc_1},e1)
				local e3 = lm.qt:backward({rep1,rep2},e2[2])
				local e4 = lm.qf:backward({rep1,rep3},e2[1])
				
				local e5 = lm.qst:backward(vecs[1],(e3[1]+e4[1])/2)
				local e7 = lm.tas:backward(vecs[2],e3[2])
				local e8 = lm.fas:backward(vecs[3],e4[2])
				
				--lm.emd:backward(sent_index[1],e5)
				--lm.emd:backward(sent_index[2],e7)
				--lm.emd:backward(sent_index[3],e8)
				
			end
			grad_params = grad_params:div(size)
			--lm.emd.gradWeight:div(size*3)
			loss = loss / size
			loss = loss + 0.5*1e-4*params:norm()^2
			return loss,grad_params		
		end
		optim.adagrad(feval,params,optim_state)
		--lm.emd:updateParameters(0.05)
	end
end
------------------------------------------------------------------------
function test_one_pair(qst_rep,answer)
	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为未经处理的句子
--[
	local lm = cfg.lm
	local ans_idx = get_index(answer):clone()
	if cfg.gpu then ans_idx = ans_idx:cuda() end
	local ans_emd = lm.emd:forward(ans_idx):clone()
	local ans_rep = lm.tas:forward(ans_emd)
	local distance = lm.qt:forward({qst_rep,ans_rep})
	return 1 - distance[1]
--]
end

function evaluate(name)
	--评估训练好的模型的精度，top 1是正确答案的比例
	local test_set = deep_cqa.insurance[name]
	local answer_set = deep_cqa.insurance['answer']
	if(test_set == nil) then print('测试集载入为空！') return end
	local lm = cfg.lm	--语言模型
	local results = {}
	print('test process:')
	--local test_count = 100
	for i,v in pairs(test_set) do
		--test_count = test_count -1
		--if test_count ==0 then break end
		xlua.progress(i,1000)

		local golden = v[1]	--正确答案的集合
		local qst = v[2]	--问题
		local candidates = v[3] --候选的答案
		
		local qst_idx = get_index(qst):clone()
		if cfg.gpu then qst_idx = qst_idx:cuda() end
		local qemd = lm.emd:forward(qst_idx):clone()
		local qst_vec = lm.qst:forward(qemd)
		
		local sc = {}	
		local golden_sc ={}
		local golden_rank = {}
		
		for k,c in pairs(golden) do 
			c =tostring(tonumber(c))
			local score = test_one_pair(qst_vec,answer_set[c])	--标准答案的得分
			golden_sc[k] = score
			golden_rank[k] = 1	--初始化排名
		end
		thr = 20	--500个样本只测前thr个
		for k,c in pairs(candidates) do 
			thr = thr -1
			if thr ==0 then break end

			c =tostring(tonumber(c))
			local score = test_one_pair(qst_vec,answer_set[c])
			for m,n in pairs(golden_sc) do
				if score > n then
					golden_rank[m] = golden_rank[m]+1
				end
			end
		end
		
		local mark =0.0
		local mrr = 0
		for k,c in pairs(golden_rank) do
			if c==1 then 
				mark = 1.0
			end
			mrr = mrr + 1.0/c
		end
		results[i] = {mrr,1.0}

	end
	local results = torch.Tensor(results)
	print(torch.sum(results,1)/results:size()[1])
end
--getlm()
--testlm()
train()
evaluate('dev')

