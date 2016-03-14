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
	self.bw = torch.load('bilstm.weight')
end
-----------------------
function CoSim:get_score(question, answer,mark)
	--简单计算答案相对问题的相似度得分
	local qst = string.gsub(question,'\r',''):split(' ')
	local ans = string.gsub(answer,'\r',''):split(' ')
	local score = 0
	local s2 ={}
	for i,qw in pairs(qst) do
		local s1 =0;
--[
		local bqw = self.bw[qw]
		if bqw ~= nil then
			bqw = bqw[1]/bqw[2]
		else
			bqw = 1
		end
--]
		for j,aw in pairs(ans) do
--[
			local baw = self.bw[aw]
			if baw ~= nil then
				baw = baw[1]/baw[2]
			else
				baw = 1
			end
--]
			if self.co_matrix[qw]~=nil and self.co_matrix[qw][aw]~=nil then
				s3 = (self.co_matrix[qw][aw])*bqw*baw
				--s3 =  self.co_matrix[qw][aw]
				if s3>1 then
				s1 = s1+s3
				end
	--			print(qw,bqw,aw,baw,self.co_matrix[qw][aw]*bqw*baw)
			else
				s1 = s1 * 1
			end
		end
		score = score + s1/#ans
		s2[qst[i]]= s1/#ans
	end	
	if mark==true then
	--	print(s2)
	end
--	print(score)		
	return score
end
