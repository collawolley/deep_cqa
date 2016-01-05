--[[
	通过文本的共现度，计算问题和答案之间的相似度
	可能用到的数据：
	截止词
	共现矩阵（从训练集中提取得来）
	idf统计结果
--]]
local CoSim = torch.class('CoSim')
function CoSim:__init()		
	-- 初始化，载入截止词和共现矩阵
	self.sw = torch.load(deep_cqa.config.stop_words)
	self.wc = torch.load(deep_cqa.config.word_count)
	self.co_matrix = torch.load(deep_cqa.config.co_matrix,'binary')
end
-----------------------
function CoSim:get_score(question, answer)
	--简单计算答案相对问题的相似度得分
	local qst = string.gsub(question,'\r',''):split(' ')
	local ans = string.gsub(answer,'\r',''):split(' ')
	local score = {}
	for i,aw in pairs(ans) do
		score[i] = 0
		if self.sw[aw]== nil then
			for j,qw in pairs(qst) do
				if self.sw[qw] == nil then
					local t = self.co_matrix[qw]
					if t ~= nil then 
						t = t[aw]				
					end
					if t == nil then
						t =0
					else
						t=t/(self.wc[aw][1]*self.wc[qw][1])
						--t=t/(self.wc[aw][1])
					end
					score[i] = score[i]+t				
				end
			end
		end
	end
	local value =0.0 -- 返回给调用代码的相似度值
	for i,v in pairs(score) do 
		value = value + v
	end
	value = value/#score
	return value
end
--[[
tmp = CoSim()
sc = tmp:get_score('today is a good day','tomorrw is a good day too')
print(sc)
--]]
