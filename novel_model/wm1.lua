--[[
	给词向量自己加权重
	author:	liangjz
	time:	2015-01-25
--]]
--------------------
local  WM = torch.class('WM1')
function WM: __init(useGPU)
	self.cfg = {	--配置文件
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	= deep_cqa.config.emd_dim,	--词向量的维度
		cuda= useGPU or false,	--是否使用gpu模式
		margin	= 0.009,
		l2	= 0.0001,	--L2范式的约束
		lr = 0.01	--L2范式的约束
	}	
	self.cfg.dict, self.cfg.vecs = deep_cqa.get_sub_embedding()
	self.cfg.emd = nn.LookupTable(self.cfg.vecs:size(1),self.cfg.dim)
	self.cfg.emd.weight:copy(self.cfg.vecs)
	self.cfg.vecs = nil
	self.LM = {}	--语言模型
	self:getLM()	--生成语言模型
	self.dataSet = InsSet()	--保险数据集，这里载入是为了获得测试集和答案
end	
-----------------------

function WM:getIndex(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	return deep_cqa.read_one_sentence(sent,self.cfg.dict)
end
-----------------------

function WM:getLM()	--获取语言模型
	self.LM ={}	--清空原始的模型
	local lm = self.LM
	local cfg = self.cfg
	lm.emd = {}	--三组同步的词向量
	lm.conv = {}	--卷积层
	lm.wm = {}	--对词向量进行重新加权
	lm.rep = {}	--获得统一表达
	lm.cosine = {}	--cosine 相似度计算结果
	
	local pt = nn.Sequential()
	pt:add(nn.SpatialAdaptiveMaxPooling(1,1))
	pt:add(nn.Reshape(1000))
	pt:add(nn.Tanh())

	local reweight = nn.Sequential()
	local c1 = nn.ConcatTable()
	c1:add(nn.LinearNoBias(self.cfg.dim,1))
	c1:add(nn.Identity())
	reweight:add(nn.CMul(self.cfg.dim))

	for i = 1,3 do
		lm.emd[i] = self.cfg.emd:clone()
		lm.conv[i] = nn.SpatialConvolution(1,1000,self.cfg.dim,2,1,1,0,1)	--input需要是3维tensor
		lm.wm[i] = reweight:clone()
		lm.rep[i] = nn.Sequential()
		lm.rep[i]:add(lm.wm[i])
		lm.rep[i]:add(nn.Tanh())
		lm.rep[i]:add(nn.Replicate(1))
		lm.rep[i]:add(lm.conv[i])
		lm.rep[i]:add(pt:clone())
	end
	lm.cosine[1] = nn.CosineDistance()	--nn包里的cosine distance实际上计算方式为wiki的cosine similarity
	lm.cosine[2] = nn.CosineDistance()	--nn包里的cosine distance实际上计算方式为wiki的cosine similarity
	lm.sub = nn.CSubTable()

	if self.cfg.cuda then
		for i =1,3 do
			lm.emd[i]:cuda()
			lm.rep[i]:cuda()
		end
		lm.cosine[1]:cuda()
		lm.cosine[2]:cuda()
		lm.sub:cuda()
	end
	for i=2,3 do
		lm.emd[i]:share(lm.emd[1],'weight','bias')
		lm.conv[i]:share(lm.conv[1],'weight','bias','gradWeight','gradBias')
		lm.wm[i]:share(lm.wm[1],'weight','bias','gradWeight','gradBias')
	end
	self.LM.sub:zeroGradParameters()
	self.LM.cosine[1]:zeroGradParameters()
	self.LM.cosine[2]:zeroGradParameters()
	for i = 1,3 do
		self.LM.emd[i]:zeroGradParameters()
		self.LM.rep[i]:zeroGradParameters()
	end
	local p,g
	p,g = lm.wm[1]:parameters()
	for i,v in pairs(p) do
		v[1]= 1
	end

end
------------------------

function WM:testLM()	--应用修改模型后测试模型是否按照预期执行
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1})
	local index = {}
	index[1] = self:getIndex('today is a good day'):clone()
	index[2] = self:getIndex('today is a very good day'):clone()
	index[3] = self:getIndex('This class creates an output where the input is replicated'):clone()
	local vec = self.cfg.emd:forward(index[1])
	local reweight = nn.Sequential()
	local c1 = nn.ConcatTable()
	c1:add(nn.LinearNoBias(self.cfg.dim,1))
	c1:add(nn.Identity())
	reweight:add(nn.CMul(self.cfg.dim))
	print(reweight:forward(vec))
end
--------------------------
function WM:train(negativeSize)
	self.dataSet:resetTrainset(negativeSize)	--设置训练集中每次选取负样本的数量
	local modules = nn.Parallel()
		for i =1,3 do
			modules:add(self.LM.rep[i])
		end
	params,grad_params = modules:getParameters()
	for i=2,3 do
		self.LM.emd[i]:share(self.LM.emd[1],'weight','bias')
		self.LM.conv[i]:share(self.LM.conv[1],'weight','bias','gradWeight','gradBias')
		self.LM.wm[i]:share(self.LM.wm[1],'weight','bias','gradWeight','gradBias')
	end
	local criterion = nn.MarginCriterion(self.cfg.margin)
	local gold = torch.Tensor({1})
	if self.cfg.cuda then
		criterion:cuda()
		gold = gold:cuda()
	end
	local sample =1	--占个坑
	local index = {}	--存储字符下标
	local vecs={}	--存储词向量
	local rep ={}	--存储句子的定长向量表达
	local cosine ={}	--存储两个句子之间的相似度
	local loop = 0
	local right_sample = 0

	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
		if loop %1 ==0 then xlua.progress(self.dataSet.current_train,self.dataSet.train_set.size) end
		sample = self.dataSet:getNextPair()
		if sample == nil then break end	--数据集获取完毕
		for i = 1,3 do
			index[i] = self:getIndex(sample[i]):clone()
		end
		if loop %2==0 then
			index[2],index[3] = index[3],index[2]
			gold[1]= -1
		else
			gold[1] =1
		end
		for i =1,3 do
			if self.cfg.cuda then
				index[i] = index[i]:cuda()
			end
			vecs[i] = self.LM.emd[i]:forward(index[i]):clone()
			rep[i] = self.LM.rep[i]:forward(vecs[i])
		end
		cosine[1] = self.LM.cosine[1]:forward({rep[1],rep[2]})
		cosine[2] = self.LM.cosine[2]:forward({rep[1],rep[3]})
		local pred = self.LM.sub:forward(cosine)

		local err = criterion:forward(pred,gold)
		if err <= 0 then
			right_sample = right_sample + 1
		end

		local e0 = criterion:backward(pred,gold)
		e1 = e0  + self.cfg.l2*0.5*params:norm()^2	--二阶范
		local esub= self.LM.sub:backward(cosine,e1)
		local ecosine = {}
		ecosine[1] = self.LM.cosine[1]:backward({rep[1],rep[2]},esub[1])
		ecosine[2] = self.LM.cosine[2]:backward({rep[1],rep[3]},esub[2])
		local erep = {}
		local erep1 = {}
		erep1[1] = (ecosine[1][1] + ecosine[2][1])/2
		erep1[2] = ecosine[1][2]
		erep1[3] = ecosine[2][2]
		for  i = 1,3 do
			erep[i] = self.LM.rep[i]:backward(vecs[i],erep1[i])
			self.LM.emd[i]:backward(index[i],erep[i])
		end
		local lr  = self.cfg.lr
		self.LM.sub:updateParameters(lr)
		self.LM.sub:zeroGradParameters()
		self.LM.cosine[1]:updateParameters(lr)
		self.LM.cosine[1]:zeroGradParameters()
		self.LM.cosine[2]:updateParameters(lr)
		self.LM.cosine[2]:zeroGradParameters()

		for i = 1,3 do
			self.LM.emd[i]:updateParameters(lr)
			self.LM.emd[i]:zeroGradParameters()
			self.LM.rep[i]:updateParameters(lr)
			self.LM.rep[i]:zeroGradParameters()
		end
	end
	print('训练集的准确率：',right_sample/loop)
end
-------------------------

function WM:testOnePair(question_vec,answer_id)	--传入的qst为已经计算好的向量，ans为问题的id
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	local sim_sc = self.LM.cosine[1]:forward({question_vec,answer_rep})
	return sim_sc[1]
end

function WM:evaluate(name)	--评估训练好的模型的精度，top 1是正确答案的比例
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
		if self.cfg.cuda then word_index = word_index:cuda() end
		local answer_emd = self.LM.emd[2]:forward(word_index):clone()
		local answer_rep = self.LM.rep[2]:forward(answer_emd):clone()
		self.dataSet:saveAnswerVec(answer_pair[1],answer_rep)
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
		if self.cfg.cuda then qst_idx = qst_idx:cuda() end
		local qst_emd = self.LM.emd[1]:forward(qst_idx):clone()
		local qst_vec = self.LM.rep[1]:forward(qst_emd):clone()

		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			local score = self:testOnePair(qst_vec,c)	--标准答案的得分,传入内容为问题的表达和答案的编号
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end

		for k,c in pairs(candidates) do 
			local score = self:testOnePair(qst_vec,c)
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
	print('Results:',torch.sum(results,1)/results:size()[1])
end
