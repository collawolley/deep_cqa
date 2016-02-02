--[[
	coco数据集的warper，用类的形式提供服务
	liangjz 2015-1-31
	正样本为语义相同的句子，负样本为与该正样本最相似的句子
--]]
local Coco = torch.class('CocoSet')	--保险语料库的读取warper
function Coco:__init(negativeSize)
	self.corp = torch.load(deep_cqa.config.coco.corp)
	self.trainSize = 20000 --#self.corp.trainid	--82783最多
	if self.trainSize > 82783 then self.trainSize = 82783 end
	self.valSize  = 1000
	self.testSize = 1000
	if self.valSize+self.testSize > 30000 then self.valSize = 10000 self.testSize = 20000 end
	self.indices = torch.randperm(#self.corp.trainid)

	self.current_train = 1	--当前样本的下标
	self.current_answer = 2	--当前正确答案的下标，一个问题可能会有多个正确答案 2-5
	self.current_negative = 1	--当前选取的负样本为第多少个负样本（针对这个训练样本来说）

	self.negativeSize =negativeSize or 1	--负样本采样个数
	if self.negativeSize  > 10 then self.nengativeSize = 10 end

	self.answer_vecs ={}	--存储答案的中间结果
	self.answer_index = 1	--遍历答案时的下标
	self.dev_index = 1	--遍历验证集时下标
	self.test_index = 1	--遍历测试集时的下标
end

function Coco:getNextPair()	--生成下一对问题-正样本-负样本对
	if self.current_train > self.trainSize then return nil end	--数据集已经遍历完毕
	local array = self.corp.trainid[self.current_train]
	local qst = self.corp.sents[array[1]]--一个问题
	local true_id = array[self.current_answer]	--一个正确答案
	local false_id = self.corp.nearSents[true_id][self.current_negative]
	local false_answer = self.corp.sents[false_id]
	local true_answer = self.corp.sents[true_id]

	self.current_negative = self.current_negative + 1
	if self.current_negative > self.negativeSize then
		self.current_negative = 1
		self.current_answer = self.current_answer + 1
	end
	if self.current_answer > #array then	--当前问题的正确回答已经遍历完毕
		self.current_answer = 2
		self.current_train = self.current_train + 1
	end
	return {qst,true_answer,false_answer}	--返回sample三元组
end

function Coco:resetTrainset(negativeSize)	--重新设定训练集
	self.indices = torch.randperm(#self.corp.trainid)	--对训练样本进行乱序处理
	self.current_train = 1	--当前样本的下标
	self.current_answer = 2	--当前正确答案的下标，一个问题可能会有多个正确答案
	self.current_negative = 1	--当前选取的负样本为第多少个负样本（针对这个训练样本来说）
	self.negativeSize =negativeSize or self.negativeSize	--负样本采样个数
	if self.negativeSize  > 10 then self.nengativeSize = 10 end
end

function Coco:getAnswer(answer_id)	--根据编号返回答案的文本内容
	return self.corp.sents[answer_id]
end

function Coco:getNextAnswer(marker)	--返回下一个编号的答案
	local reset = (marker == true) or false	--是否从头开始
	if reset then self.answer_index = 1 end
	if self.answer_index > #self.corp.sents then return nil end	--超出下标则返回空答案
	self.answer_index = self.answer_index + 1
	return {self.answer_index-1,self.corp.sents[self.answer_index-1]}	--返回答案下标和答案内容的二元组
end

function Coco:saveAnswerVec(answer_id,answer_vec)	--保存答案的向量表达形式
	self.answer_vecs[answer_id] = answer_vec	
end

function Coco:getAnswerVec(answer_id)	--读取答案的向量表达形式
	return self.answer_vecs[answer_id]
end

function Coco:getNextDev(marker)	--获取下一个验证集样本	【正确ids，问题，候选ids】
	local reset = (marker == true) or false	--是否从头开始
	if reset then self.dev_index = 1 end
	if self.dev_index > self.valSize then return nil end	--超出下标则返回空验证组
	self.dev_index = self.dev_index + 1
	local true_id = self.corp.valid[self.dev_index-1][2]
	local qst_id = self.corp.valid[self.dev_index-1][1]
	local candidate = {}
	for i =1,9 do 
		candidate[i] = self.corp.nearSents[true_id][i]
	end
	return {true_id,qst_id,candidate}	--返回验证集的三元组
end

function Coco:getNextTest(marker)	--获取下一个测试集样本
	local reset = (marker == true) or false	--是否从头开始
	if reset then self.test_index = 1 end

	if self.test_index > self.testSize then return nil end	--超出下标则返回空答案
	self.test_index = self.test_index + 1 
	local true_id = self.corp.valid[self.valSize-1+self.test_index][2]
	local qst_id = self.corp.valid[self.valSize-1+self.test_index][1]
	local candidate = {}
	for i =1,9 do 
		candidate[i] = self.corp.nearSents[true_id][i]
	end
	return {true_id,qst_id,candidate}	--返回验证集的三元组
end

