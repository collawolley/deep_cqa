--[[
	原来的使用的训练方法可能是错误的，现在在训练过程中，先训练一个样本使其收敛后再训练其他样本
	author:	liangjz
	time:	2015-01-11
--]]
require('..')
local cfg = {}
cfg.vecs = nil
cfg.dict = nil	--字典
cfg.emd = nil	--词向量
cfg.dim = deep_cqa.config.emd_dim	--词向量的维度
cfg.gpu = true	--是否使用gpu模式
cfg.L2Rate =0.0001	--L2范式的约束
data_set = InsSet()	--保险数据集，这里载入是为了获得测试集和答案
cfg.margin = 0.009
-----------------------
function get_index(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	if cfg.dict == nil then --	载入字典和词向量查询层
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
	local qcov = nn.SpatialConvolution(1,1000,200,2,1,1,0,1)	--input需要是3维tensor
	local tcov = qcov:clone('weight','bias')
	local fcov = qcov:clone('weight','bias')
-------------------------------------
	local pt = nn.Sequential()
	pt:add(nn.SpatialAdaptiveMaxPooling(1,1))
	pt:add(nn.Reshape(1000))
	pt:add(nn.Tanh())
-------------------------------------
	local hlq = nn.Linear(cfg.dim,200)
	local hlt = hlq:clone('weight','bias')
	local hlf = hlq:clone('weight','bias')
-------------------------------------
	local lm = {}	--待返回的语言模型
	lm.qemd = cfg.emd	--词嵌入部分
	lm.temd = lm.qemd:clone('weight','bias')
	lm.femd = lm.qemd:clone('weight','bias')
	lm.qst = nn.Sequential()
	lm.tas = nn.Sequential()
	lm.fas = nn.Sequential()

	lm.qst:add(hlq)
	lm.tas:add(hlt)
	lm.fas:add(hlf)
	---------------------
	lm.qst:add(nn.Tanh())
	lm.tas:add(nn.Tanh())
	lm.fas:add(nn.Tanh())

	---------------------
	lm.qst:add(nn.Replicate(1))
	lm.tas:add(nn.Replicate(1))
	lm.fas:add(nn.Replicate(1))
	---------------------
	lm.qst:add(qcov)
	lm.tas:add(tcov)
	lm.fas:add(fcov)
	---------------------
	lm.qst:add(pt:clone())
	lm.tas:add(pt:clone())
	lm.fas:add(pt:clone())
	----------------------
	lm.qt = nn.CosineDistance()	--nn包里的cosine distance实际上计算方式为wiki的cosine similarity
	lm.qf = nn.CosineDistance()
--	lm.sub = nn.PairwiseDistance(1)
	lm.sub = nn.CSubTable()
-------------------------------
	if cfg.gpu then
		lm.qemd:cuda()
		lm.temd:cuda()
		lm.femd:cuda()
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
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1})
	
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
	
	local hl = nn.Linear(cfg.dim,200)
	local q =  lm.qst:forward(vec1)
	local t =  lm.tas:forward(vec2)
	local f =  lm.fas:forward(vec3)
	local qt = lm.qt:forward({q,t})
	local qf = lm.qf:forward({q,f})
	local sub = lm.sub:forward({qf,qt})
	print(qt,qf,sub)
end
--------------------------
cfg.lm = getlm()
-------------------------
function train()
	local lm = cfg.lm
	local modules = nn.Parallel()
	modules:add(lm.qst)
	modules:add(lm.tas)
	modules:add(lm.fas)
	modules:add(lm.tq)
	modules:add(lm.fq)
	modules:add(lm.sub)
	params,grad_params = modules:getParameters()
	lm.qst:share(lm.tas,'weight','bias','gradWeight','gradBias')
	lm.fas:share(lm.tas,'weight','bias','gradWeight','gradBias')

	local criterion = nn.MarginCriterion(cfg.margin)
	local gold = torch.Tensor({1})
	if cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	local learningRate = 0.01
	--local learningRate = 0.03

	local next_sample = true	--是否获取下一个sample
	local sample =1	--占个坑
	local vecs={}	--存储词向量
	local index ={}	--存储字符下标
	local loop = 1

	local sample_count = 0
	local right_sample = 0
	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
		if loop %100 ==0 then xlua.progress(data_set.current_train,data_set.train_set.size) end
		if next_sample then
			sample = data_set:getNextPair()
			if sample == nil then break end	--数据集获取完毕
			index[1] = get_index(sample[1]):clone()
			index[2] = get_index(sample[2]):clone()
			index[3] = get_index(sample[3]):clone()
			if(cfg.gpu) then
				index[1] = index[1]:cuda() 
			 	index[2] = index[2]:cuda() 
			 	index[3]= index[3]:cuda() 
			end
			next_sample = true --满足特定条件才获取下一个sample
		end
--[
		if loop % 2 ==0 then
			index[2],index[3] = index[3],index[2]
			gold[1] = -1
		else
			gold[1] = 1
		end
--]
		vecs[1] = lm.qemd:forward(index[1]):clone()
		vecs[2] = lm.temd:forward(index[2]):clone()
		vecs[3] = lm.femd:forward(index[3]):clone()	
		
		local rep1 = lm.qst:forward(vecs[1])
		local rep2 = lm.tas:forward(vecs[2])
		local rep3 = lm.fas:forward(vecs[3])
				
		local sc_1 = lm.qt:forward({rep1,rep2})
		local sc_2 = lm.qf:forward({rep1,rep3})
		local pred = lm.sub:forward({sc_1,sc_2})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
		
		local err = criterion:forward(pred,gold)
		--print(pred[1],err)
		sample_count = sample_count + 1
		if err <= 0 then
			next_sample = true
			right_sample = right_sample + 1
		end
	--	else
			lm.sub:zeroGradParameters()
			lm.qt:zeroGradParameters()
			lm.qf:zeroGradParameters()
			lm.qst:zeroGradParameters()
			lm.tas:zeroGradParameters()
			lm.fas:zeroGradParameters()
			lm.qemd:zeroGradParameters()
			lm.temd:zeroGradParameters()
			lm.femd:zeroGradParameters()				
			local e0 = criterion:backward(pred,gold)
			e1 = e0  + cfg.L2Rate*0.5*params:norm(2)	--二阶范数
			--print('loss',pred[1],err,e0[1],e1[1])
			local e2 = lm.sub:backward({sc_1,sc_2},e1)
			local e3 = lm.qt:backward({rep1,rep2},e2[1])
			local e4 = lm.qf:backward({rep1,rep3},e2[2])
			
			local e5 = lm.qst:backward(vecs[1],(e3[1]+e4[1])/2)
			local e7 = lm.tas:backward(vecs[2],e3[2])
			local e8 = lm.fas:backward(vecs[3],e4[2])
			
			lm.sub:updateParameters(learningRate)
			lm.qt:updateParameters(learningRate)
			lm.qf:updateParameters(learningRate)
			lm.qst:updateParameters(learningRate)
			lm.tas:updateParameters(learningRate)
			lm.fas:updateParameters(learningRate)
--[	
			lm.qemd:backward(index[1],e5)
			lm.qemd:updateParameters(learningRate)
			lm.temd:backward(index[2],e7)
			lm.temd:updateParameters(learningRate)
			lm.femd:backward(index[3],e8)
			lm.femd:updateParameters(learningRate)
--]
--		end
	end
	print('\n训练集的准确率：',right_sample/sample_count)
end
------------------------------------------------------------------------
function test_one_pair(question_vec,answer_id) 	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为问题的id
	local lm = cfg.lm
	local answer_rep = data_set:getAnswerVec(answer_id)	--获取答案的表达
	if cfg.gpu then
		answer_rep = answer_rep:cuda()
	end
	local sim_sc = lm.qt:forward({question_vec,answer_rep})
--	print(question_vec[1],answer_id,answer_rep[1],sim_sc[1])
	return sim_sc[1]
end
function evaluate(name)	--评估训练好的模型的精度，top 1是正确答案的比例
	local lm = cfg.lm	--语言模型
	local results = {}
	local test_size = 0
	local loop = 0
	if name =='dev' then 
		test_size = data_set.dev_set.size 
	else
		test_size = data_set.test_set.size 
	end

--	data_set.answer_vecs = torch.load('model/answer_vecs','binary')
--[
	print('\nCalculating answers')
	local answer_pair = data_set:getNextAnswer(true)	--从头开始计算answer的向量
	while answer_pair~=nil do
		loop = loop+1
		xlua.progress(loop,data_set.answer_set.size)
		local answer = answer_pair[2]	--获取问题内容
		local word_index = get_index(answer)	--获取词下标
		if cfg.gpu then word_index = word_index:cuda() end
		local answer_emd = lm.temd:forward(word_index):clone()
		local answer_rep = lm.tas:forward(answer_emd):clone()
		--print(answer,word_index[1],answer_rep[1])
		data_set:saveAnswerVec(answer_pair[1],answer_rep)
		answer_pair = data_set:getNextAnswer()
	end	
--	torch.save('model/answer_vecs',data_set.answer_vecs,'binary')
--]
	collectgarbage() 
	print('\nTest process:')
	local test_pair =nil
	if name =='dev' then
		test_pair = data_set:getNextDev(true)
	else
		test_pair = data_set:getNextTest(true)
	end
	loop = 0
	while test_pair~=nil do
		loop = loop+1
		xlua.progress(loop,test_size)

		local gold = test_pair[1]	--正确答案的集合
		local qst = test_pair[2]	--问题
		local candidates = test_pair[3] --候选的答案
		local qst_idx = get_index(qst)
		if cfg.gpu then qst_idx = qst_idx:cuda() end
		local qst_emd = lm.qemd:forward(qst_idx):clone()
		local qst_vec = lm.qst:forward(qst_emd):clone()

		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			local score = test_one_pair(qst_vec,c)	--标准答案的得分,传入内容为问题的表达和答案的编号
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end
		--print('\n',torch.Tensor(gold_sc))

		for k,c in pairs(candidates) do 
			local score = test_one_pair(qst_vec,c)
		--	print(score)
			for m,n in pairs(gold_sc) do
				if score > n then
					gold_rank[m] = gold_rank[m]+1
				end
			end
		end
		
		local mark =0.0
		local mrr = 0
		for k,c in pairs(gold_rank) do
			if c==1 then 
				mark = 1.0
			end
			mrr = mrr + 1.0/c
		end
		results[loop] = {mrr,mark}
		if name =='dev' then
			test_pair = data_set:getNextDev()
		else
			test_pair = data_set:getNextTest()
		end

		if loop%10==0 then collectgarbage() end
	end

	local results = torch.Tensor(results)
--	print(results)
	print('\nResults:',torch.sum(results,1)/results:size()[1])
end
--getlm()
--testlm()
--train()
--torch.save('model/cov_sdg2_1.bin',cfg.lm,'binary')
--data_set:resetTrainset(1)
--cfg.lm = torch.load('model/cov_1.bin','binary')
--evaluate('dev')

--cfg.lm = torch.load('model/cov_sdg2_lc9_20.bin','binary')
--evaluate('dev')
for epoch =1,50 do 
	print('\nTraining in ',epoch,'epoch:')
	cfg.L2Rate = 0.0001--0.0001*3^epoch
	--local margin={0.003,0.009,0.03,0.1,0.3,1}
--	local l2={0.0003,0.01,0.03,0.1,0.3,1}
--	cfg.dict = nil
--	cfg.lm ={}
--	cfg.lm = getlm()
	data_set:resetTrainset(50)
	if epoch >1 then
		data_set:resetTrainset(2)
	end
	cfg.margin = 0.009
	cfg.L2Rate = 0.0001
	print('L2Rate:',cfg.L2Rate)
	print('Margin:',cfg.margin)
	train()
	--cfg.lm = torch.load('model/cov_sdg2_lc2_' .. epoch ..'.bin','binary')
--	torch.save('model/cov_sdg2_lc10_' .. epoch ..'.bin',cfg.lm,'binary')
	evaluate('dev')
--	cfg.margin = cfg.margin*3
end

