require('..')
---------------------------


function main( ... )
	local vec_dict,vecs = deep_cqa.get_sub_embedding()
	local emd_layer = nn.LookupTable(vecs:size(1),deep_cqa.config.emd_dim)
	emd_layer.weight:copy(vecs)
	
end
---------------------------
main()
