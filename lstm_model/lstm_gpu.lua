--[[
	使用rnn的LSTM代码实现正确答案和错误答案的分类	
	author:	liangjz
	time:	2015-12-22
--]]
require('..')
local cfg = {}
cfg.vecs = nil
cfg.dict = nil
cfg.emd = nil
cfg.dim = deep_cqa.config.emd_dim
cfg.mem = 30
-----------------------
function get_index(sent)
	--	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	if cfg.dict == nil then
		--	载入字典和词向量查询层
		cfg.dict, cfg.vecs = deep_cqa.get_sub_embedding()
		cfg.emd = nn.LookupTable(cfg.vecs:size(1),cfg.dim)
		cfg.emd.weight:copy(cfg.vecs)
		cfg.vecs = nil
	end
	return deep_cqa.read_one_sentence(sent,cfg.dict)
end
--------------------------
function get_emd(index)
	--	备用函数，输入词的index组，输出查询到的词向量
	if cfg.emd == nil then 
		get_index('good start')
	end
	return  cfg.emd:forward(index):cuda()
end
-------------------------
function sim_model()
	--	先测试一下LSTM的速度
	local qst= nn.Identity()()
	local tans= nn.Identity()()
	local fans= nn.Identity()()
	---------------------------
	local qlstm = nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem))(qst)
	local tlstm = nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem))(tans)
	local flstm = nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem))(fans)
	share_params (tlstm,flstm)
	---------------------------
	local qsel = nn.SelectTable(-1)(qlstm)
	local tsel = nn.SelectTable(-1)(tlstm)
	local fsel = nn.SelectTable(-1)(flstm)
	---------------------------
	local rsp_qst = nn.Reshape(cfg.dim,1)(qsel)
	local rsp_tans = nn.Reshape(1,cfg.mem)(tsel)
	local rsp_fans = nn.Reshape(1,cfg.mem)(fsel)
	
	local tm =nn.MM()({rsp_qst,rsp_tans})
	local fm =nn.MM()({rsp_qst,rsp_fans})

	local sub = nn.CSubTable()({tm,fm})
	local norm = nn.SoftSign()(sub)
	local reshape = nn.Reshape(cfg.mem*cfg.mem)(norm)
	local linear = nn.Linear(cfg.mem*cfg.mem,1)(reshape)
	local model = nn.gModule({qst,tans,fans},{linear})
	
	return model:cuda()
end

function train()
	local model =sim_model()
--	local modules = nn.Parallel():add(qst_lstm):add(tans_lstm):add(fans_lstm):add(model)
-------------------
	params,grad_params = model:getParameters()

	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1}):cuda()
	local batch_size = 10
	local optim_state = {learningRate = 0.05 }
	for i= 1,train_set.size,batch_size do
		local size = math.min(i+batch_size-1,batch_size)-i+1
		
		local feval = function(x)
			grad_params:zero()
			local loss = 0	--累积的损失
			
			for j = 1,size do
				xlua.progress(i+j-1,train_set.size)
				local idx = indices[i+j-1]
				local sample = train_set[idx]
				local vecs={}
				for k =1,#sample do
					local index = get_index(sample[k])
					vecs[k] = get_emd(index)
					vecs[k] = vecs[k]:chunk(vecs[k]:size(1),1)
				end

				if(idx %2 ==0) then
					vecs[3],vecs[2] = vecs[2],vecs[3]
					gold[1] = -1
				else
					gold[1] = 1
				end
				local pred = model:forward(vecs)
				local loss = loss + criterion:forward(pred,gold)
				
				local obj_grad = criterion:backward(pred,gold)
				local lstm_grad = model:backward(vecs,obj_grad:cuda())
				
			end
			grad_params = grad_params/size
			loss = loss / size
			loss = loss + 1e-4*params:norm()^2
			return loss,grad_params		
		end
		optim.adagrad(feval,params,optim_state)
	end
end
train()
