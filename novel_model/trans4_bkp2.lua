--[[
	给词向量的表达加tf-idf
	author:	liangjz
	time:	2015-01-19
--]]
--------------------
local Trans4 = torch.class('Trans4')
function Trans4: __init(useGPU)
	self.cfg = {	--配置文件
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	= deep_cqa.config.emd_dim,	--词向量的维度
		gpu	= useGPU or false,	--是否使用gpu模式
		margin	= 0.009,
		l2Rate	= 0.0001,	--L2范式的约束
		learningRate	= 0.01	--L2范式的约束
	}	
	self.cfg.dict, self.cfg.vecs = deep_cqa.get_sub_embedding()
	self.cfg.emd = nn.LookupTable(self.cfg.vecs:size(1),self.cfg.dim)
	self.cfg.emd.weight:copy(self.cfg.vecs)
	self.cfg.vecs = nil
	self.LM = {}	--语言模型
	self:getLM()	--生成语言模型
	self.dataSet = InsSet()	--保险数据集，这里载入是为了获得测试集和答案
	self.wc = torch.load(deep_cqa.config.word_count)
end	
-----------------------

function Trans4:getIndex(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	return deep_cqa.read_one_sentence(sent,self.cfg.dict)
end
-----------------------

function Trans4:getLM()	--获取语言模型
	self.LM ={}	--清空原始的模型
	local tcov = nn.SpatialConvolution(1,1000,200,2,1,1,0,1)	--input需要是3维tensor
	local fcov = tcov:clone('weight','bias')
	local qcov = tcov:clone('weight','bias')
-------------------------------------
	local pt = nn.Sequential()
	pt:add(nn.SpatialAdaptiveMaxPooling(1,1))
	pt:add(nn.Reshape(1000))
	pt:add(nn.Tanh())
-------------------------------------
	local hlt =  nn.Linear(self.cfg.dim,200)
	local hlf = hlt:clone('weight','bias')
	local hlq = hlt:clone('weight','bias')
-------------------------------------
	self.LM.qemd = self.cfg.emd	--词嵌入部分
	self.LM.temd = self.LM.qemd:clone('weight','bias')
	self.LM.femd = self.LM.qemd:clone('weight','bias')

	self.LM.qst1 = nn.Sequential()
	self.LM.tas1 = nn.Sequential()
	self.LM.fas1 = nn.Sequential()
	
	self.LM.qst2 = nn.Sequential()
	self.LM.tas2 = nn.Sequential()
	self.LM.fas2 = nn.Sequential()
	
	self.LM.qm = nn.MM()
	self.LM.tm = nn.MM()
	self.LM.fm = nn.MM()

	self.LM.qst1:add(hlq)
	self.LM.tas1:add(hlt)
	self.LM.fas1:add(hlf)
	---------------------
	self.LM.qst1:add(nn.Tanh())
	self.LM.tas1:add(nn.Tanh())
	self.LM.fas1:add(nn.Tanh())
	---------------------
	self.LM.qst2:add(nn.Replicate(1))
	self.LM.tas2:add(nn.Replicate(1))
	self.LM.fas2:add(nn.Replicate(1))
	---------------------
	self.LM.qst2:add(qcov)
	self.LM.tas2:add(tcov)
	self.LM.fas2:add(fcov)
	---------------------
	self.LM.qst2:add(pt:clone())
	self.LM.tas2:add(pt:clone())
	self.LM.fas2:add(pt:clone())
	----------------------
	self.LM.qt = nn.CosineDistance()	--nn包里的cosine distance实际上计算方式为wiki的cosine similarity
	self.LM.qf = nn.CosineDistance()
	self.LM.sub = nn.CSubTable()
	----------------------
	if self.cfg.gpu then
		self.LM.qemd:cuda()
		self.LM.temd:cuda()
		self.LM.femd:cuda()
		self.LM.qst1:cuda()
		self.LM.tas1:cuda()
		self.LM.fas1:cuda()
		self.LM.qst2:cuda()
		self.LM.tas2:cuda()
		self.LM.fas2:cuda()
		self.LM.qt:cuda()
		self.LM.qf:cuda()
		self.LM.sub:cuda()

		self.LM.qm:cuda()
		self.LM.tm:cuda()
		self.LM.fm:cuda()
	end
end
------------------------

function Trans4:testLM()	--应用修改模型后测试模型是否按照预期执行
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1})
	local index1 = self.getIndex('today is a good day'):clone()
	local index2 = self.getIndex('today is a very good day'):clone()
	local index3 = self.getIndex('This class creates an output where the input is replicated'):clone()
	if cfg.gpu then
		criterion:cuda()
		gold =gold:cuda()
		index1 = index1:cuda()
		index2 = index2:cuda()
		index3 = index3:cuda()
	end
	local vec1 = self.LM.qemd:forward(index1):clone()
	local vec2 = self.LM.temd:forward(index2):clone()
	local vec3 = self.LM.femd:forward(index3):clone()
	
	local hl = nn.Linear(cfg.dim,200)
	local q =  self.LM.qst:forward(vec1)
	local t =  self.LM.tas:forward(vec2)
	local f =  self.LM.fas:forward(vec3)
	local qt = self.LM.qt:forward({q,t})
	local qf = self.LM.qf:forward({q,f})
	local sub = self.LM.sub:forward({qf,qt})
	print(qt,qf,sub)
end
--------------------------
function Trans4:getWeight(indcs)	--获取对角矩阵，对应元素为权重
	local sgmd = nn.Tanh()
	local size = indcs:size()[1]
	local M = torch.Tensor(size,size):zero()
	for i =1,size do
		local id = indcs[i]
		local w = self.cfg.dict:token(id)
		local score = 0
		if  self.wc[w]==nil then
			score = 0
		else
			score = self.wc[w][1]
		end
		if score ==1 then score =0 end
		score = sgmd:forward(torch.Tensor({score}))[1]
		--print(w,score)
		score =1+score
		M[i][i]=  score
	end
	--print (M)
	return M
	
end

function Trans4:train(negativeSize)
	self.dataSet:resetTrainset(negativeSize)	--设置训练集中每次选取负样本的数量
	local modules = nn.Parallel()
	modules:add(self.LM.qst1)
	modules:add(self.LM.tas1)
	modules:add(self.LM.fas1)
	modules:add(self.LM.qst2)
	modules:add(self.LM.tas2)
	modules:add(self.LM.fas2)
	modules:add(self.LM.qt)
	modules:add(self.LM.qf)
	modules:add(self.LM.sub)
	params,grad_params = modules:getParameters()
	
	local criterion = nn.MarginCriterion(self.cfg.margin)
	local gold = torch.Tensor({1})
	if self.cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	self.LM.temd:share(self.LM.qemd,'weight','bias')
	self.LM.femd:share(self.LM.qemd,'weight','bias')
	self.LM.tas1:share(self.LM.qst1,'weight','bias','gradWeight','gradBias')
	self.LM.fas1:share(self.LM.qst1,'weight','bias','gradWeight','gradBias')
	self.LM.tas2:share(self.LM.qst2,'weight','bias','gradWeight','gradBias')
	self.LM.fas2:share(self.LM.qst2,'weight','bias','gradWeight','gradBias')
	local sample =1	--占个坑
	local vecs={}	--存储词向量
	local index ={}	--存储字符下标
	local wws = {}	--word weights
	local loop = 0

	local sample_count = 0
	local right_sample = 0
	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
		if loop %100 ==0 then xlua.progress(self.dataSet.current_train,self.dataSet.train_set.size) end
		sample = self.dataSet:getNextPair()
		if sample == nil then break end	--数据集获取完毕
		index[1] = self:getIndex(sample[1]):clone()
		index[2] = self:getIndex(sample[2]):clone()
		index[3] = self:getIndex(sample[3]):clone()
		if loop %2==0 then 
			gold[1]=-1
			index[2],index[3] = index[3],index[2]
        else
			gold[1]=1
		end
		wws = {}
		for i=1,3 do
			wws[i] = self:getWeight(index[i]:clone()):clone()
		end
		if(self.cfg.gpu) then
			index[1] = index[1]:cuda() 
		 	index[2] = index[2]:cuda() 
		 	index[3]= index[3]:cuda() 

			wws[1]= wws[1]:cuda()
			wws[2]= wws[2]:cuda()
			wws[3]= wws[3]:cuda()
		end
		vecs[1] = self.LM.qemd:forward(index[1]):clone()
		vecs[2] = self.LM.temd:forward(index[2]):clone()
		vecs[3] = self.LM.femd:forward(index[3]):clone()	
		
		local rep1 = self.LM.qst1:forward(vecs[1])
		local rep2 = self.LM.tas1:forward(vecs[2])
		local rep3 = self.LM.fas1:forward(vecs[3])
		local rep4 = self.LM.qm:forward({wws[1],rep1})
		local rep5 = self.LM.tm:forward({wws[2],rep2})
		local rep6 = self.LM.fm:forward({wws[3],rep3})

		local rep7 = self.LM.qst2:forward(rep4)
		local rep8 = self.LM.tas2:forward(rep5)
		local rep9 = self.LM.fas2:forward(rep6)
		


		local sc_1 = self.LM.qt:forward({rep7,rep8})
		local sc_2 = self.LM.qf:forward({rep7,rep9})
		local pred = self.LM.sub:forward({sc_1,sc_2})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
		--print(sc_1[1],sc_2[1],pred[1])
		--pred[1] = pred[1]  + self.cfg.l2Rate*0.5*params:norm()^2	--二阶范
		local err = criterion:forward(pred,gold)
		sample_count = sample_count + 1
		if err <= 0 then
			right_sample = right_sample + 1
		end
		self.LM.sub:zeroGradParameters()
		self.LM.qt:zeroGradParameters()
		self.LM.qf:zeroGradParameters()
		self.LM.qst1:zeroGradParameters()
		self.LM.tas1:zeroGradParameters()
		self.LM.fas1:zeroGradParameters()
		self.LM.qst2:zeroGradParameters()
		self.LM.tas2:zeroGradParameters()
		self.LM.fas2:zeroGradParameters()
		self.LM.qm:zeroGradParameters()
		self.LM.tm:zeroGradParameters()
		self.LM.fm:zeroGradParameters()
		self.LM.qemd:zeroGradParameters()
		self.LM.temd:zeroGradParameters()
		self.LM.femd:zeroGradParameters()				

		local e0 = criterion:backward(pred,gold)
		e1 = e0  + self.cfg.l2Rate*0.5*params:norm()^2	--二阶范
		local e2 = self.LM.sub:backward({sc_1,sc_2},e1)
		local e3 = self.LM.qt:backward({rep7,rep8},e2[1])
		local e4 = self.LM.qf:backward({rep7,rep9},e2[2])
		
		local e5 = self.LM.qst2:backward(rep4,(e3[1]+e4[1])/2)
		local e6 = self.LM.tas2:backward(rep5,e3[2])
		local e7 = self.LM.fas2:backward(rep6,e4[2])

		local e8 = self.LM.qm:backward({wws[1],rep1},e5)
		local e9 = self.LM.tm:backward({wws[2],rep2},e6)
		local e10 = self.LM.fm:backward({wws[3],rep3},e7)
		
		local e11 = self.LM.qst1:backward(vecs[1],e8[2])
		local e12 = self.LM.tas1:backward(vecs[2],e9[2])
		local e13 = self.LM.fas1:backward(vecs[3],e10[2])
		
		local learningRate  = self.cfg.learningRate
		self.LM.sub:updateParameters(learningRate)
		self.LM.qt:updateParameters(learningRate)
		self.LM.qf:updateParameters(learningRate)

		self.LM.qst1:updateParameters(learningRate)
		self.LM.tas1:updateParameters(learningRate)
		self.LM.fas1:updateParameters(learningRate)
		
		self.LM.qst2:updateParameters(learningRate)
		self.LM.tas2:updateParameters(learningRate)
		self.LM.fas2:updateParameters(learningRate)

		self.LM.qemd:backward(index[1],e11)
		self.LM.qemd:updateParameters(learningRate)
		self.LM.temd:backward(index[2],e12)
		self.LM.temd:updateParameters(learningRate)
		self.LM.femd:backward(index[3],e13)
		self.LM.femd:updateParameters(learningRate)
	end
	print('训练集的准确率：',right_sample/sample_count)
end
-------------------------

function Trans4:test_one_pair(question_vec,answer_id) 	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为问题的id
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	--print('ans_rep',answer_id,answer_rep[1][1])
	if self.cfg.gpu then
		answer_rep = answer_rep:cuda()
	end
	local sim_sc = self.LM.qt:forward({question_vec,answer_rep})
	return sim_sc[1]
end

function Trans4:evaluate(name)	--评估训练好的模型的精度，top 1是正确答案的比例
	local results = {}	--记录每个测试样本的结果
	local test_size = 0
	local loop = 0
	if name =='dev' then 
		test_size = self.dataSet.dev_set.size 
	else
		test_size = self.dataSet.test_set.size 
	end
	print('Calculating answers')
	local answer_pair =self.dataSet:getNextAnswer(true)	--从头开始计算answer的向量
	while answer_pair~=nil do
		loop = loop+1
		xlua.progress(loop,self.dataSet.answer_set.size)
		local answer = answer_pair[2]	--获取问题内容
		local word_index = self:getIndex(answer)	--获取词下标
		local ww = self:getWeight(word_index):clone()
		if self.cfg.gpu then word_index = word_index:cuda() ww= ww:cuda() end
		local answer_emd = self.LM.temd:forward(word_index):clone()
		local answer_rep1 = self.LM.tas1:forward(answer_emd):clone()
		local answer_rep2 = self.LM.tm:forward({ww,answer_rep1}):clone()
		local answer_rep3 = self.LM.tas2:forward(answer_rep2):clone()
		self.dataSet:saveAnswerVec(answer_pair[1],answer_rep3)
		answer_pair = self.dataSet:getNextAnswer()
	end	
	collectgarbage() 
	print('Test process:')
	local test_pair =nil
	if name =='dev' then
		test_pair = self.dataSet:getNextDev(true)
	else
		test_pair = self.dataSet:getNextTest(true)
	end
	loop = 0
	while test_pair~=nil do
		loop = loop+1
		xlua.progress(loop,test_size)

		local gold = test_pair[1]	--正确答案的集合
		local qst = test_pair[2]	--问题
		local candidates = test_pair[3] --候选的答案
		local qst_idx = self:getIndex(qst)
		local qww = self:getWeight(qst_idx):clone()
		if self.cfg.gpu then qst_idx = qst_idx:cuda() qww = qww:cuda() end
		local qst_emd = self.LM.qemd:forward(qst_idx):clone()
		local qst_vec1 = self.LM.qst1:forward(qst_emd):clone()
		local qst_vec2 = self.LM.qm:forward({qww,qst_vec1}):clone()
		local qst_vec = self.LM.qst2:forward(qst_vec2):clone()

		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			local score = self:test_one_pair(qst_vec,c)	--标准答案的得分,传入内容为问题的表达和答案的编号
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end

		for k,c in pairs(candidates) do 
			local score = self:test_one_pair(qst_vec,c)
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
			test_pair = self.dataSet:getNextDev()
		else
			test_pair = self.dataSet:getNextTest()
		end

		if loop%10==0 then collectgarbage() end
	end

	local results = torch.Tensor(results)
	print('\nResults:',torch.sum(results,1)/results:size()[1])
end

