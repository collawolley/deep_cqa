--[[
	给词向量的表达加tf-idf
	author:	liangjz
	time:	2015-01-19
	mdifiy: 2016-2-3
--]]
--------------------
local Trans = torch.class('Trans4')
function Trans: __init(useGPU)
	self.cfg = {	--配置文件
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	= deep_cqa.config.emd_dim,	--词向量的维度
		gpu	= useGPU or false,	--是否使用gpu模式
		margin	= 0.009,
		l2	= 0.0001,	--L2范式的约束
		lr	= 0.01	--L2范式的约束
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
	local lm = self.LM
	lm.conv = {}
	lm.linear = {}
	lm.emd = {}
	lm.mm = {}
	lm.rep1 = {}
	lm.rep2 = {}
	lm.cosine = {}
	lm.sub = nil
	local tcov = nn.SpatialConvolution(1,1000,200,2,1,1,0,1)	--input需要是3维tensor
	local fcov = tcov:clone('weight','bias')
	local qcov = tcov:clone('weight','bias')
-------------------------------------
	local pt = nn.Sequential()
	pt:add(nn.SpatialAdaptiveMaxPooling(1,1))
	pt:add(nn.Reshape(1000))
	pt:add(nn.Tanh())
-------------------------------------
	for i =1,3 do 
		lm.emd[i] = self.cfg.emd:clone()
		lm.linear[i] = nn.Linear(self.cfg.dim,200)
		lm.conv[i] = nn.SpatialConvolution(1,1000,200,2,1,1,0,1)	--input需要是3维tensor
		lm.mm[i] = nn.MM()
		lm.rep1[i] = nn.Sequential()
		lm.rep2[i] = nn.Sequential()	
		lm.rep1[i]:add(lm.linear[i])
		lm.rep1[i]:add(nn.Tanh())
		lm.rep2[i]:add(nn.Replicate(1))
		lm.rep2[i]:add(lm.conv[i])
		lm.rep2[i]:add(pt:clone())
	end
	----------------------
	lm.cosine[1] = nn.CosineDistance()	--nn包里的cosine distance实际上计算方式为wiki的cosine similarity
	lm.cosine[2] = nn.CosineDistance()
	lm.sub = nn.CSubTable()
	----------------------
	if self.cfg.gpu then
		for i =1,3 do
			lm.emd[i]:cuda()
			lm.mm[i]:cuda()
			lm.rep1[i]:cuda()
			lm.rep2[i]:cuda()
		end
		lm.cosine[1]:cuda()
		lm.cosine[2]:cuda()
		lm.sub:cuda()
	end
end
------------------------

function Trans4:testLM()	--应用修改模型后测试模型是否按照预期执行
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1})
	local index1 = self.getIndex('today is a good day'):clone()
	local index2 = self.getIndex('today is a very good day'):clone()
	local index3 = self.getIndex('This class creates an output where the input is replicated'):clone()
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
	for i =1,3 do
		modules:add(self.LM.rep1[i])
		modules:add(self.LM.rep2[i])
	end
	params,grad_params = modules:getParameters()
	for i =2,3 do
		self.LM.emd[i]:share(self.LM.emd[1],'weight','bias')
		self.LM.rep1[i]:share(self.LM.rep1[1],'weight','bias','gradWeight','gradBias')
		self.LM.rep2[i]:share(self.LM.rep2[1],'weight','bias','gradWeight','gradBias')
	end
	local criterion = nn.MarginCriterion(self.cfg.margin)
	local gold = torch.Tensor({1})
	if self.cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end

	local sample =1	--占个坑
	local vecs={}	--存储词向量
	local index ={}	--存储字符下标
	local wws = {}	--word weights
	local rep1 = {}
	local rep2 = {}
	local rep3 = {}
	local cos = {}
	local loop = 0

	local right_sample = 0
	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
		if loop %100 ==0 then xlua.progress(self.dataSet.current_train,self.dataSet.train_set.size) end
		sample = self.dataSet:getNextPair()
		if sample == nil then break end	--数据集获取完毕
		for i =1,3 do 
			index[i] = self:getIndex(sample[i]):clone()
		end
		if loop %2==0 then 
			gold[1]=-1
			index[2],index[3] = index[3],index[2]
        else
			gold[1]=1
		end

		wws = {}
		for i=1,3 do
			wws[i] = self:getWeight(index[i]:clone()):clone()
			if self.cfg.gpu then
				index[i] = index[i]:cuda() 
				wws[i] = wws[i]:cuda()
			end
			vecs[i] = self.LM.emd[i]:forward(index[i]):clone()
			rep1[i] = self.LM.rep1[i]:forward(vecs[i])
			rep2[i] = self.LM.mm[i]:forward({wws[i],rep1[i]})
			rep3[i] = self.LM.rep2[i]:forward(rep2[i])
		end
		cos[1] = self.LM.cosine[1]:forward({rep3[1],rep3[2]})
		cos[2] = self.LM.cosine[2]:forward({rep3[1],rep3[3]})
		local pred = self.LM.sub:forward(cos)
		local err = criterion:forward(pred,gold)
		if err <= 0 then
			right_sample = right_sample + 1
		end
		self.LM.sub:zeroGradParameters()
		self.LM.cosine[1]:zeroGradParameters()
		self.LM.cosine[2]:zeroGradParameters()
		for i  = 1,3 do
			self.LM.rep1[i]:zeroGradParameters()
			self.LM.rep2[i]:zeroGradParameters()
			self.LM.mm[i]:zeroGradParameters()
			self.LM.emd[i]:zeroGradParameters()
		end
		local e0 = criterion:backward(pred,gold)
		e1 = e0  + self.cfg.l2*0.5*params:norm()^2	--二阶范
		local esub = self.LM.sub:backward(cos,e1)
		local ecos = {}
		ecos[1] = self.LM.cosine[1]:backward({rep3[1],rep3[2]},esub[1])
		ecos[2] = self.LM.cosine[2]:backward({rep3[1],rep3[3]},esub[2])
		local erep2 = {}
		local emm = {}
		local erep1 = {}
		erep2[1] = self.LM.rep2[1]:backward(rep2[1],(ecos[1][1]+ecos[2][1])/2)
		erep2[2] = self.LM.rep2[2]:backward(rep2[2],ecos[1][2])
		erep2[3] = self.LM.rep2[3]:backward(rep2[3],ecos[2][2])
		for i =1,3 do
			emm[i] = self.LM.mm[i]:backward({wws[i],rep1[i]},erep2[i])
			erep1[i] = self.LM.rep1[i]:backward(vecs[i],emm[i][2])
			self.LM.emd[i]:backward(index[i],erep1[i])
		end
		local lr  = self.cfg.lr
		self.LM.sub:updateParameters(lr)
		self.LM.cosine[1]:updateParameters(lr)
		self.LM.cosine[1]:zeroGradParameters()
		self.LM.cosine[2]:updateParameters(lr)
		self.LM.cosine[2]:zeroGradParameters()
		for i =1 ,3 do
			self.LM.rep1[i]:updateParameters(lr)
			self.LM.rep1[i]:zeroGradParameters()
			self.LM.rep2[i]:updateParameters(lr)
			self.LM.rep2[i]:zeroGradParameters()
			self.LM.emd[i]:updateParameters(lr)
			self.LM.emd[i]:zeroGradParameters()
			self.LM.mm[i]:updateParameters(lr)
			self.LM.mm[i]:zeroGradParameters()
		end
	end
	print('训练集的准确率：',right_sample/loop)
end
-------------------------

function Trans4:test_one_pair(question_vec,answer_id) --传入的qst为已经计算好的向量，ans为问题的id
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	local sim_sc = self.LM.cosine[1]:forward({question_vec,answer_rep})
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
		local answer_emd = self.LM.emd[2]:forward(word_index):clone()
		local answer_rep1 = self.LM.rep1[2]:forward(answer_emd):clone()
		local answer_rep2 = self.LM.mm[2]:forward({ww,answer_rep1}):clone()
		local answer_rep3 = self.LM.rep2[2]:forward(answer_rep2):clone()
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
		local qst_emd = self.LM.emd[1]:forward(qst_idx):clone()
		local qst_vec1 = self.LM.rep1[1]:forward(qst_emd):clone()
		local qst_vec2 = self.LM.mm[1]:forward({qww,qst_vec1}):clone()
		local qst_vec = self.LM.rep2[1]:forward(qst_vec2):clone()

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

