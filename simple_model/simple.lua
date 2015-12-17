require('..')
---------------------------
function main( ... )
	local vec_dict,vecs = deep_cqa.get_sub_embedding()
	local emd_layer = nn.LookupTable(vecs:size(1),deep_cqa.config.emd_dim)
	emd_layer.weight:copy(vecs)

	local sents = deep_cqa.read_sentences(deep_cqa.config.train_corpus , vec_dict)
	local idx = sents[1]
	local vecs = emd_layer:forward(sents[1])
	print('idx',idx)
	print('vecs',vecs)
	local avg_emd = deep_cqa.AvgEmd()
	local ans = avg_emd:forward(vecs)
	print('ans',ans)
end
---------------------------
function get_model()
--[[
	--词嵌入以及权值加成的计算
	local vec_dict,vecs = deep_cqa.get_sub_embedding()
	deep_cqa.emd_dict =	vec_dict
	deep_cqa.emd_vecs = vecs
	local emd_layer = nn.LookupTable(vecs:size(1),deep_cqa.config.emd_dim)
	emd_layer.weight:copy(vecs)
	local avg_emd = deep_cqa.AvgEmd()
--]]
	--------------------------
	--			margin 误差统计			-->	误差计算criterion
	--			diff(1)作差				--> 共享层（误差是否能够传递通过这层）
	--	sigmoid()			sigmoid()
	--	线性(1)				线性(1)		--> 共享权重
	--	并行(600)			并行(600)	--> 共享权重
	local qst = nn.Identity()()		--问题的向量均值表达（也可以是前一层采用了RNN之类的方法实现） 300维
	local t_ans = nn.Identity()()	--正确答案的向量表达（同上） 300维
	local f_ans = nn.Identity()()	--错误。。。（同上）
	local inputs  = {qst,t_ans.f_ans}
	
	local t_rep = nn.JoinTable(1){qst,t_ans}	--两个向量做拼接，600维
	local f_rep = nn.JoinTable(1){qst,f_ans}	--同上

	local t_prob = nn.Linear(600,1)(t_rep)	--转换成为预测是否为正确答案的标量	
	local f_prob = nn.Linear(600,1)(f_rep)	--转换成为预测是否为错误答案的标量
	
	t_prob = nn.gModule({qst,t_ans},t_prob)
	f_prob = nn.gModule({qst,f_ans},f_prob)
	share_params(t_prob,f_prob)

	local t_sig = nn.Sigmoid()(t_prob)	--加一个sigmoid函数，约束值的范围
	local f_sig = nn.Sigmoid()(f_prob)

	local sub  = nn.CSubTable{t_sig,f_sig}	--作差，优化的目标是使这个差逼近一个margin
	local simple = nn.gModule(inputs,nn.Identity()(sub))
	
	return simple
end
---------------------------
function train()
	local simple = get_model()
	local criterion = nn.BCECriterion()	--误差计算
	local batch_size = deep_cqa.config.batch_size
end
---------------------------
Simple = {}
Simple.vecs = nil
Simple.dict = nil
Simple.emd_layer = nil
Simple.avg =nil
function get300(sent)
	if Simple.vecs == nil then
		Simple.dict,Simple.vecs = deep_cqa.get_sub_embedding()
		Simple.emd_layer = nn.LookupTable(vecs:size(1),deep_cqa.config.emd_dim)
		Simple.emd_layer.weight:copy(vecs)
		Simple.avg = deep_cqa.AvgEmd()
	end
	local idx = deep_cqa.read_one_sentence(sent,Simple.dict)
	local vecs = Simple.emd_layer:forward(idx)
	local ans = Simple.avg:forward(vecs)
	
	return ans
end

function demo()
	local sents = {'the is a simple text file','how is the sun today','what happens last night'}
	local simple = get_model()
	local vecs = {}
	for i =1,#sents do
		vecs[i] = get300(sents[i])
	end
	simple:forward(vecs)
end
demo()
