--项目的引用的包
require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
deep_cqa = {}
-------------------------------
--项目的目录配置部分
config = {}

config.parent_path = lfs.currentdir() .. '/'
config.emd_vec = config.parent_path .. 'data/glove/glove.840B.300d.th'
config.emd_dict = config.parent_path .. 'data/glove/glove.840B.vocab'
config.train_corpus = config.parent_path .. 'data/text.txt'
config.dict = config.parent_path .. 'data/text.dict'

config.emd_dim = 300
config.batch_size = 10

deep_cqa.config = config
-------------------------------
include('util/vocab.lua')
include('util/emd.lua')
include('util/read_data.lua')
include('simple_model/avg_emd.lua')

