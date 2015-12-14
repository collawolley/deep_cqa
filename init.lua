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
config.emd_vec = 'data/glove/glove.840B.300d.th'
config.emd_dict = 'data/glove/glove.840B.vocab'
config.emd_dim = 300
config.train_corpus = 'data/text.txt'
config.dict = 'data/text.dict'

deep_cqa.config = config
-------------------------------
--include('Vocab.lua')
--include('emd.lua')
