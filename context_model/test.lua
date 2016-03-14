--[[
	测试基于共现的方法是否有一定精度（效果是否好于随机）
	liangjz 2015-1-4
--]]
require('..')
dataSet = InsSet(1)
function test(name)
	cosim = CoSim()
	local results = {}
	print('test process:')
	local test_pair = dataSet:getNextDev(true)
	loop = 0
	while test_pair~=nil do
		loop = loop + 1
		xlua.progress(loop,1000)

		local golden = test_pair[1]	--正确答案的集合
		local qst = test_pair[2]	--问题
		local candidates = test_pair[3] --候选的答案
		
		local sc = {}	
		local golden_sc ={}
		local golden_rank = {}
	--	print(qst)
		for k,c in pairs(golden) do 
			c =tostring(tonumber(c))
			local score = cosim:get_score(qst,dataSet:getAnswer(c),true)	--标准答案的得分
			golden_sc[k] = score
			golden_rank[k] = 1	--初始化排名
		end
	--	print('other answers:')
		for k,c in pairs(candidates) do 
			c =tostring(tonumber(c))
			local score = cosim:get_score(qst,dataSet:getAnswer(c))
			for m,n in pairs(golden_sc) do
				if score > n then
					golden_rank[m] = golden_rank[m]+1
	--				print('Over')
					cosim:get_score(qst,dataSet:getAnswer(c),true)
				end
			end
		end
		local mark =0.0
		local mrr = 0.0
		for k,c in pairs(golden_rank) do
			if c==1 then 
				mark = 1.0
			end
			mrr = mrr + 1.0/c
		end
		results[loop] = {mrr,mark}
		if loop%50 ==0 then collectgarbage() end
		test_pair = dataSet:getNextDev()
	end
	local results = torch.Tensor(results)
	print(torch.sum(results,1)/results:size()[1])

end
test('dev')
