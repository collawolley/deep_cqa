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
	--	线性(1)				线性(1)	--> 共享权重
	--	并行(600)			并行(600)	--> 共享权重
	
	local inputs  = nn.Identity()()

	local simple = nn.Sequential()
	local merg = nn.Parallel(1,1)
	
	local ta_part = nn.Sequential()
	local fa_part = nn.Sequential()
	
	local ta_emd = nn.Parallel(1,1)
	local ta_a = nn.Sequential()
	ta_a:add(emd_layer)
	ta_a:add(avg_emd)
	local ta_q = nn.Sequential()
	ta_q:add(emd_layer)
	ta_a:add(avg_emd)
	ta_emd:add(ta_q)
	ta_emd:add(ta_a)
	ta_part:add(ta_emd)	--到这一步获得了一个600维的向量表示句子
	ta_part:add(nn.Linear(600,1))
	ta_part:add(nn.Sigmoid())
	
	local fa_emd = nn.Parallel(1,1)
	local fa_a = nn.Sequential()
	fa_a:add(emd_layer)
	fa_a:add(avg_emd)
	local fa_q = nn.Sequential()
	fa_q:add(emd_layer)
	fa_a:add(avg_emd)
	fa_emd:add(fa_q)
	fa_emd:add(fa_a)
	fa_part:add(fa_emd)	--到这一步获得了一个600维的向量表示句子
	fa_part:add(nn.Linear(600,1))
	fa_part:add(nn.Sigmoid())
	
	local merg = nn.CSubTable(){ta_part,fa_part}


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
