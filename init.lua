--项目的引用的包
require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('rnn')
require('cunn')
deep_cqa = {}
-------------------------------
--项目的目录配置部分
config = {}
-------------------------------
config.parent_path = lfs.currentdir() .. '/'
--------词向量配置
--[	--性能最好的词向量
config.emd_vec = config.parent_path .. 'data/glove/glove.840B.300d.th'
config.emd_dict = config.parent_path .. 'data/glove/glove.840B.vocab'
config.emd_dim = 300
--]
-------
--[[
config.emd_vec = config.parent_path .. 'data/word2vec/Google_300.vec'
config.emd_dict = config.parent_path .. 'data/word2vec/Google_300.dict'
config.emd_dim = 300
--]]
-----
--[[
config.emd_vec = config.parent_path .. 'data/word2vec/SG_10.vec'
config.emd_dict = config.parent_path .. 'data/word2vec/SG_10.dict'
config.emd_dim = 10
--]]
------
--[[
config.emd_vec = config.parent_path .. 'data/word2vec/SG_30.vec'
config.emd_dict = config.parent_path .. 'data/word2vec/SG_30.dict'
config.emd_dim = 30
--]]
------
--[[
config.emd_vec = config.parent_path .. 'data/word2vec/SG_50.vec'
config.emd_dict = config.parent_path .. 'data/word2vec/SG_50.dict'
config.emd_dim = 50
--]]
------
--[[
config.emd_vec = config.parent_path .. 'data/word2vec/ins_100.vec'
config.emd_dict = config.parent_path .. 'data/word2vec/ins_100.dict'
config.emd_dim = 100
--]]
----- 
--[[	--这个组词向量的性能不如glove的词向量性能好
config.emd_vec = config.parent_path .. 'data/word2vec/mix_100.vec'
config.emd_dict = config.parent_path .. 'data/word2vec/mix_100.dict'
config.emd_dim = 100
--]]
-----------
config.insurance = {}
config.insurance.train = config.parent_path .. 'data/insurance_qa/' .. 'train.txt'
config.insurance.dev = config.parent_path .. 'data/insurance_qa/' .. 'dev.txt'
config.insurance.test1 = config.parent_path .. 'data/insurance_qa/' .. 'test1.txt'
config.insurance.test2 = config.parent_path .. 'data/insurance_qa/' .. 'test2.txt'
config.insurance.answer = config.parent_path .. 'data/insurance_qa/' .. 'answer.txt'
config.insurance.dict = config.parent_path .. 'data/insurance_qa/' .. 'dict.txt'
config.insurance.binary = config.parent_path .. 'data/insurance_qa/' .. 'full_dataset.bin'
config.insurance.negative_size =1
-------------
config.batch_size = 5
config.random_seed =134
config.stop_words = 'data/context/stop_words.tab'
config.co_matrix = 'data/context/co_matrix.tab'
config.word_count = 'data/context/word_count.tab'
-------------------------
deep_cqa.config = config
deep_cqa.insurance = {}
deep_cqa.ins_meth ={}
deep_cqa.ins_meth.train = config.parent_path .. 'data/insurance_qa/' .. 'train_1.bin'
deep_cqa.ins_meth.validation = config.parent_path .. 'data/insurance_qa/' .. 'train_10.bin'
deep_cqa.ins_meth.test = config.parent_path .. 'data/insurance_qa/' .. 'train_10.bin'
-------------------------------
include('util/vocab.lua')
include('util/emd.lua')
include('util/read_data.lua')
include('simple_model/avg_emd.lua')
include('context_model/co_sim.lua')


--deep_cqa.ins_meth.load_txt_dataset()
--deep_cqa.ins_meth.save_binary()

--deep_cqa.ins_meth.load_binary()
--deep_cqa.ins_meth.generate_train_set()

function share_params(cell, src)
	if torch.type(cell) == 'nn.gModule' then
		for i = 1, #cell.forwardnodes do
			local node = cell.forwardnodes[i]
			if node.data.module then
				node.data.module:share(src.forwardnodes[i].data.module,'weight', 'bias', 'gradWeight', 'gradBias')
			end
		end
	elseif torch.isTypeOf(cell, 'nn.Module') then
		cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
	elseif torch.type(cell) == 'nngraph.Node' then
		cell.data.module:share(src.data.module, 'weight', 'bias', 'gradWeight', 'gradBias')
	else
		print(torch.type(cell))
		error('parameters cannot be shared for this input')
	end
end

function share_weights(cell, src)	--仅仅共享权重，导数不共享，这样在多个模块共享参数的时候，不影响各自求导
	if torch.type(cell) == 'nn.gModule' then
		for i = 1, #cell.forwardnodes do
			local node = cell.forwardnodes[i]
			if node.data.module then
				node.data.module:share(src.forwardnodes[i].data.module,'weight', 'bias')
			end
		end
	elseif torch.isTypeOf(cell, 'nn.Module') then
		cell:share(src, 'weight', 'bias')
	elseif torch.type(cell) == 'nngraph.Node' then
		cell.data.module:share(src.data.module, 'weight', 'bias')
	else
		print(torch.type(cell))
		error('parameters cannot be shared for this input')
	end
end
