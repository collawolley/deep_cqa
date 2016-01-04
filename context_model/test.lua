--[[
	测试基于共现的方法是否有一定精度（效果是否好于随机）
	liangjz 2015-1-4
--]]
require('..')
deep_cqa.ins_meth.load_binary()	
function test(name)
	cosim = CoSim()
	local test_set = deep_cqa.insurance[name]
	local answer_set = deep_cqa.insurance['answer']
	if(test_set == nil) then print('测试集载入为空！') return end
	
	local results = {}
	print('test process:')
	for i,v in pairs(test_set) do
		xlua.progress(i,1000)

		local golden = v[1]	--正确答案的集合
		local qst = v[2]	--问题
		local candidates = v[3] --候选的答案
		
		local sc = {}	
		local golden_sc ={}
		local golden_rank = {}
		
		for k,c in pairs(golden) do 
			c =tostring(tonumber(c))
			local score = cosim:get_score(qst,answer_set[c])	--标准答案的得分
			golden_sc[k] = score
			golden_rank[k] = 1	--初始化排名
		end
		for k,c in pairs(candidates) do 
			c =tostring(tonumber(c))
			local score = cosim:get_score(qst,answer_set[c])
			for m,n in pairs(golden_sc) do
				if score > n then
					golden_rank[m] = golden_rank[m]+1
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
		results[i] = {mrr,mark}
		if i%50 ==0 then collectgarbage() end
	--	if i>99 then break end
	end
	local results = torch.Tensor(results)
	print(torch.sum(results,1)/results:size()[1])

end
test('dev')
