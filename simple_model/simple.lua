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
	local vec_dict,vecs = deep_cqa.get_sub_embedding()
	deep_cqa.emd_dict =	vec_dict
	deep_cqa.emd_vecs = vecs
	local emd_layer = nn.LookupTable(vecs:size(1),deep_cqa.config.emd_dim)
	emd_layer.weight:copy(vecs)
	local avg_emd = deep_cqa.AvgEmd()
	
	local simple = nn.Sequential()
	local parall = nn.Parallel(2,1)
	--------------------------------
	local a = nn.Sequential()
	a:add(emd_layer)
	a:add(deep_cqa.AvgEmd())
	local q = nn.Sequential()
	q:add(emd_layer)
	q:add(deep_cqa.AvgEmd())
	--------------------------------
	parall:add(a)	--答案的向量生成
	parall:add(q)	--问题的向量生成
	simple:add(parall)	--二者向量拼接
	simple:add(nn.Linear(600,20))	--线性变换
	simple:add(nn.Sigmoid)			--非线性
	simple:add(nn.Linear(20,2))		--线性变化为类的预测
	simple:add(nn.Sigmoid)			--非线性变换

	--------------------------------
	return simple
end
---------------------------
function deep_cqa.get_next_corpus(size,train)


end
---------------------------
function train()
	local simple = get_model()
	local criterion = nn.BCECriterion()	--误差计算
	local batch_size = deep_cqa.config.batch_size


end
---------------------------
get_model()
