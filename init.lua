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
------------
config.insurance = {}
config.insurance.train = config.parent_path .. 'data/insurance_qa/' .. 'train.txt'
config.insurance.dev = config.parent_path .. 'data/insurance_qa/' .. 'dev.txt'
config.insurance.test1 = config.parent_path .. 'data/insurance_qa/' .. 'test1.txt'
config.insurance.test2 = config.parent_path .. 'data/insurance_qa/' .. 'test2.txt'
config.insurance.answer = config.parent_path .. 'data/insurance_qa/' .. 'answer.txt'
config.insurance.dict = config.parent_path .. 'data/insurance_qa/' .. 'dict.txt'
config.insurance.binary = config.parent_path .. 'data/insurance_qa/' .. 'full_dataset.bin'
config.insurance.negative_size = 100
-------------
config.emd_dim = 300
config.batch_size = 10

deep_cqa.config = config
deep_cqa.insurance = {}
-------------------------------
include('util/vocab.lua')
include('util/emd.lua')
include('util/read_data.lua')
include('simple_model/avg_emd.lua')
deep_cqa.ins_meth.load_binary()
deep_cqa.ins_meth.generate_train_set()
