--[[
	尝试实现WEC模型，并验证其是否有作用
	author:	liangjz
	time:	2015-01-15
--]]
require('..')
local cfg = {}
cfg.vecs = nil
cfg.dict = nil	--字典
cfg.emd = nil	--词向量
cfg.dim = deep_cqa.config.emd_dim	--词向量的维度
cfg.gpu = true	--是否使用gpu模式
cfg.L2Rate =0.0001	--L2范式的约束
cfg.margin = 0.009
data_set = InsSet(1)	--保险数据集，这里载入是为了获得测试集和答案
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
	local lm = {}	--待返回的语言模型
	lm.hlt = nn.LinearNoBias(cfg.dim,cfg.dim)	--转移矩阵
	lm.hlt['weight']:zero()
	for i =1,cfg.dim do
		lm.hlt['weight'][i][i] =1
	end
	lm.hlf = lm.hlt:clone('weight','bias')	--不共享权重和偏移
	lm.hlq = nn.Identity()
-------------------------------------
	--下面是cos sim的计算nn图
	local m1 = nn.ConcatTable()
	m1:add(nn.Identity())
	m1:add(nn.Identity())
	local m2 = nn.Sequential()
	m2:add(m1)
	m2:add(nn.CMulTable())
	m2:add(nn.Sum(2))
	m2:add(nn.Reshape(1))
	local m3 = m2:clone()
	local m4 = nn.ParallelTable()
	m4:add(m2)
	m4:add(m3)
	local m5 = nn.Sequential()
	m5:add(m4)
	m5:add(nn.MM(false,true))
	m5:add(nn.Sqrt())
	local m6 = nn.ConcatTable():add(nn.MM(false,true)):add(m5)
	local m7 = nn.Sequential():add(m6):add(nn.CDivTable())
	local cosine = m7
	-- cos 相似度模块构建完成
-------------------------------------

	lm.mark = lm.hlt
	lm.qemd = cfg.emd	--词嵌入部分
	lm.temd = lm.qemd:clone()
	lm.femd = lm.qemd:clone()
	lm.qst = nn.Sequential()
	lm.tas = nn.Sequential()
	lm.fas = nn.Sequential()
	lm.qst:add(lm.qemd)
	lm.tas:add(lm.temd)
	lm.fas:add(lm.femd)
	
	lm.qst:add(nn.Identity())	--问题部分的词向量不做转换
	lm.tas:add(lm.hlt)	--正负答案的部分做词向量转换
	lm.fas:add(lm.hlf)	
	lm.qst:add(nn.Tanh())	--问题部分的词向量不做转换
	lm.tas:add(nn.Tanh())	--正负答案的部分做词向量转换
	lm.fas:add(nn.Tanh())

	---------------------
	lm.qt = nn.Sequential()
	lm.qf = nn.Sequential()
	---------------------
	lm.qt:add(cosine:clone())	--计算二者的cosine相似度(问题和正样本)
	lm.qf:add(cosine:clone())	--问题和负样本
	---------------------
	lm.qt:add(nn.Max(1))	--为每个答案寻找最相近的问题中的词	
	lm.qf:add(nn.Max(1))	--缩掉问题所在的维度，为第一维
	---------------------
	lm.qt:add(nn.Mean(1))
	lm.qf:add(nn.Mean(1))
	---------------------
	lm.sub = nn.CSubTable()
-------------------------------
	if cfg.gpu then
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
	print(index1,index2,index3)
	--print(index1:size(),index2:size(),index3:size())
	local vec1 = lm.qst:forward(index1):clone()
	local vec2 = lm.tas:forward(index2):clone()
	local vec3 = lm.fas:forward(index3):clone()
	print(vec1:size(),vec2:size(),vec3:size())
	local tmp = nn.CosineDistance():cuda()
	local cos1 = lm.qt:forward({vec1,vec2})
	print(cos1)
	print(tmp:forward({vec1[2],vec2[3]}))
	print(vec1[2][1],vec2[3][1])
end
-------------------------
function train(lr)
	local lm = cfg.lm
	local modules = nn.Parallel()
	--modules:add(lm.qst)
	modules:add(lm.hlt)
	modules:add(lm.hlf)
	modules:add(lm.qt)
	modules:add(lm.qf)
	modules:add(lm.sub)
	params,grad_params = modules:getParameters()
	
	--共享参数
	lm.temd:share(lm.qemd,'weight','bias')
	lm.femd:share(lm.qemd,'weight','bias')
	lm.hlf:share(lm.hlt,'weight','bias')
	--

	local criterion = nn.MarginCriterion(cfg.margin)
	local gold = torch.Tensor({1})
	if cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	local learningRate = lr or 0.01	--LearningRate的值来自

	local next_sample = true	--是否获取下一个sample
	local sample =1	--占个坑
	local vecs={}	--存储词向量
	local index ={}	--存储字符下标
	local loop = 1

	local sample_count = 0
	local right_sample = 0
	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
        --if loop %100==0 then print(cfg.lm.mark['weight']) end
		if loop %100 ==0 then xlua.progress(data_set.current_train,data_set.train_set.size) end
		if loop%10==0 then collectgarbage() end
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

		vecs[1] = lm.qst:forward(index[1]):clone()	--问题的表达
		vecs[2] = lm.tas:forward(index[2]):clone()	--（正确）答案的表达
		vecs[3] = lm.fas:forward(index[3]):clone()	--错误答案的表达
		local qt_rep = lm.qt:forward({vecs[1],vecs[2]})	--问题和正样本最后的得分
		local qf_rep = lm.qf:forward({vecs[1],vecs[3]})	--问题和负样本最后的得分
				
		local pred = lm.sub:forward({qt_rep,qf_rep})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
		--print(qt_rep[1],qf_rep[1],pred[1])
		local err = criterion:forward(pred,gold)

		sample_count = sample_count + 1
		if err <= 0 then
			right_sample = right_sample + 1
		end

		lm.sub:zeroGradParameters()
		lm.qt:zeroGradParameters()
		lm.qf:zeroGradParameters()
		lm.qst:zeroGradParameters()
		lm.tas:zeroGradParameters()
		lm.fas:zeroGradParameters()

		local e0 = criterion:backward(pred,gold)
		e1 = e0  + cfg.L2Rate*0.5*params:norm(2)	--二阶范数
		local e2 = lm.sub:backward({qt_rep,qf_rep},e1)
		local e3 = lm.qt:backward({vecs[1],vecs[2]},e2[1])
		local e4 = lm.qf:backward({vecs[1],vecs[3]},e2[2])
			
		local e5 = lm.qst:backward(index[1],(e3[1]+e4[1])/2)
		local e7 = lm.tas:backward(index[2],e3[2])
		local e8 = lm.fas:backward(index[3],e4[2])
			
		lm.sub:updateParameters(learningRate)
		lm.qt:updateParameters(learningRate)
		lm.qf:updateParameters(learningRate)
		lm.qst:updateParameters(learningRate)
		lm.tas:updateParameters(learningRate)
		lm.fas:updateParameters(learningRate)
	end
	print('\n训练集的准确率：',right_sample/sample_count)
end
------------------------------------------------------------------------
function test_one_pair(question_vec,answer_id) 	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为问题的id	
	local lm = cfg.lm
	local answer_index = get_index(data_set:getAnswer(answer_id))
	if cfg.gpu then
		answer_index = answer_index:cuda()
	end
	local answer_rep = lm.tas:forward(answer_index)
	--local answer_rep = data_set:getAnswerVec(answer_id)	--获取答案的表达
	--if cfg.gpu then
	--	answer_rep = answer_rep:cuda()
	--end
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
--[[
	print('\nCalculating answers')
	local answer_pair = data_set:getNextAnswer(true)	--从头开始计算answer的向量
	while answer_pair~=nil do
		loop = loop+1
		if loop%10==0 then collectgarbage() end
		xlua.progress(loop,data_set.answer_set.size)
		local answer = answer_pair[2]	--获取问题内容
		local word_index = get_index(answer)	--获取词下标
		if cfg.gpu then word_index = word_index:cuda() end
		local answer_rep = lm.tas:forward(word_index):clone()
		data_set:saveAnswerVec(answer_pair[1],answer_rep)
		answer_pair = data_set:getNextAnswer()
	end	
--	torch.save('model/answer_vecs',data_set.answer_vecs,'binary')
--]]
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
		local qst_vec = lm.qst:forward(qst_idx):clone()

		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			local score = test_one_pair(qst_vec,c)	--标准答案的得分,传入内容为问题的表达和答案的编号
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end

		for k,c in pairs(candidates) do 
			local score = test_one_pair(qst_vec,c)
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
	print('\nResults:',torch.sum(results,1)/results:size()[1])
end

cfg.lm = getlm()
--testlm()
--train()
--evaluate('dev')

--[
--cfg.lm = torch.load('model/cov_sdg2_lc1_1.bin','binary')
for epoch =1,10 do
	print('\nTraining in ',epoch,'epoch:')
	--local margin={0.01,0.033,0.1,0.33,1}
--	local l2={0.0003,0.01,0.03,0.1,0.3,1}
--	cfg.dict = nil
--	cfg.lm ={}
--	cfg.lm = getlm()
	data_set:resetTrainset(1)
	cfg.margin = 0.09
	cfg.L2Rate = 0.0001
--	print('L2Rate:',cfg.L2Rate)
--	print('Margin:',cfg.margin)
	train()
	--cfg.lm = torch.load('model/cov_sdg2_lc9_' .. epoch ..'.bin','binary')
	--torch.save('model/cov_sdg2_lc8_' .. epoch ..'.bin',cfg.lm,'binary')
	evaluate('dev')
	
end
--]
