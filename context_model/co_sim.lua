--[[
	通过文本的共现度，计算问题和答案之间的相似度
	可能用到的数据：
	截止词
	共现矩阵（从训练集中提取得来）
	idf统计结果
--]]
require('math')
local CoSim = torch.class('CoSim')
function CoSim:__init()		
	-- 初始化，载入截止词和共现矩阵
	self.co_matrix = torch.load('insurance_pmi.bin')
end
-----------------------
function CoSim:get_score(question, answer)
	--简单计算答案相对问题的相似度得分
	local qst = string.gsub(question,'\r',''):split(' ')
	local ans = string.gsub(answer,'\r',''):split(' ')
	local score = 1
	local s2 ={}
	for i,qw in pairs(qst) do
		local s1 =0;
		for j,aw in pairs(ans) do
			if self.co_matrix[qw]~=nil and self.co_matrix[qw][aw]~=nil then
				s1 = s1 + math.log10(self.co_matrix[qw][aw])
--				print(qw,aw,math.log10(self.co_matrix[qw][aw]))

			else
				s1 = s1 * 1
			end
		end
		score = score + s1/#ans
		s2[i]= s1/#ans
	end
	print(s2)
--	print(score)		
	return score
end
